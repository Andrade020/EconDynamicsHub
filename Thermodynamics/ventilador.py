import cv2
import numpy as np
import random
import time
import math
import multiprocessing
import os
import functools

# ==============================================================================
# 🚀 PARÂMETROS DE ALTA PERFORMANCE
# ==============================================================================
# Agora podemos aumentar MUITO.
# 10k = 10.000 partículas por simulação (20.000 no total)
# 20k = 20.000 partículas por simulação (40.000 no total)
# Teste com 10000 primeiro. Se rodar liso, aumente para 20000 ou 50000.
TARGET_PARTICLE_COUNT = 10000
# ==============================================================================

# Constantes da Simulação
WIDTH, HEIGHT = 400, 400
ROOM_TEMP_INITIAL = 30.0
OUTSIDE_TEMP = 20.0
FAN_POSITION = 80
FAN_FORCE = 2.2
THERMAL_CONDUCTIVITY = 0.02
SPAWN_RATE = 50 # Gerar mais, pois o sistema é maior
MAX_SPEED = 5.0
PARTICLE_RADIUS = 2
INTERACTION_RADIUS = 20 # Raio de interação
INTERACTION_RADIUS_SQ = INTERACTION_RADIUS * INTERACTION_RADIUS # Pré-calcular quadrado

# Otimização: Tamanho da célula da grade = Raio de interação
CELL_SIZE = INTERACTION_RADIUS
GRID_WIDTH = math.ceil(WIDTH / CELL_SIZE)
GRID_HEIGHT = math.ceil(HEIGHT / CELL_SIZE)

# Cores (no formato BGR do OpenCV)
COLOR_HOTTEST = (0, 0, 255)
COLOR_HOT = (0, 136, 255)
COLOR_WARM = (0, 255, 136)
COLOR_COOL = (255, 170, 0)
BG_COLOR = (30, 15, 15)
FAN_COLOR = (68, 68, 68)
SENSOR_COLOR = (255, 212, 0, 0.2)
TEXT_COLOR = (255, 255, 255)

# ==============================================================================
# 1. CLASSE DA GRADE DE OTIMIZAÇÃO (SPATIAL HASH)
# ==============================================================================
class Grid:
    """Divide a tela em uma grade para achar vizinhos rapidamente."""
    def __init__(self):
        # Usamos um dicionário para "sparse grid", só armazena células que têm partículas
        self.cells = {}
        self.cell_size = CELL_SIZE

    def clear(self):
        self.cells.clear()

    def get_cell_coords(self, x, y):
        """Retorna as coordenadas (int) da célula para uma posição x, y."""
        return (int(x // self.cell_size), int(y // self.cell_size))

    def insert(self, particle):
        """Insere uma partícula na célula correta."""
        coords = self.get_cell_coords(particle.x, particle.y)
        if coords not in self.cells:
            self.cells[coords] = []
        self.cells[coords].append(particle)

    def query_neighbors(self, particle):
        """Retorna uma lista de todas as partículas vizinhas (nas 9 células ao redor)."""
        neighbors = []
        base_coords = self.get_cell_coords(particle.x, particle.y)
        
        for i in range(-1, 2): # Célula x -1, 0, +1
            for j in range(-1, 2): # Célula y -1, 0, +1
                check_coords = (base_coords[0] + i, base_coords[1] + j)
                
                if check_coords in self.cells:
                    for p in self.cells[check_coords]:
                        if p is not particle: # Não checar contra si mesmo
                            neighbors.append(p)
        return neighbors

class Particle:
    def __init__(self, x, y, temp):
        self.x = x
        self.y = y
        self.vx = (random.random() - 0.5) * 1.0
        self.vy = (random.random() - 0.5) * 1.0
        self.temp = temp
        self.is_alive = True

    def get_color(self):
        # Interpolação de cor simples (BGR)
        if self.temp >= 35: return COLOR_HOTTEST
        elif self.temp >= 28: return COLOR_HOT
        elif self.temp >= 22: return COLOR_WARM
        else: return COLOR_COOL

    def update_movement(self, fan_direction):
        """Calcula física, ventilador e paredes. (Etapa 1)"""
        if not self.is_alive: return

        # 1. Força do ventilador
        dist_to_fan = abs(self.x - FAN_POSITION)
        if dist_to_fan < 70:
            fan_strength = FAN_FORCE * (1 - dist_to_fan / 70)
            if fan_direction == 'in':
                if self.x < FAN_POSITION: self.vx += fan_strength * 2
                elif self.x < FAN_POSITION + 100: self.vx += fan_strength
            else: # 'out'
                if self.x > FAN_POSITION: self.vx -= fan_strength * 2
                elif self.x > FAN_POSITION - 100: self.vx -= fan_strength

        # 2. Movimento browniano
        thermal_agitation = math.sqrt(self.temp / 25) * 0.25
        self.vx += (random.random() - 0.5) * thermal_agitation
        self.vy += (random.random() - 0.5) * thermal_agitation

        # 3. Atrito
        self.vx *= 0.98
        self.vy *= 0.98

        # 4. Limitar velocidade
        speed_sq = self.vx**2 + self.vy**2
        if speed_sq > MAX_SPEED**2:
            speed = math.sqrt(speed_sq)
            self.vx = (self.vx / speed) * MAX_SPEED
            self.vy = (self.vy / speed) * MAX_SPEED
        
        # 5. Mover
        self.x += self.vx
        self.y += self.vy

        # 6. Colisão com paredes (top/bottom)
        if self.y < PARTICLE_RADIUS:
            self.y = PARTICLE_RADIUS
            self.vy *= -0.7
        elif self.y > HEIGHT - PARTICLE_RADIUS:
            self.y = HEIGHT - PARTICLE_RADIUS
            self.vy *= -0.7

        # 7. Checar limites (left/right) - marca para remoção
        if self.x < 0 or self.x > WIDTH:
            self.is_alive = False

    def calculate_interaction_deltas(self, grid):
        """Calcula as *mudanças* de temp/velocidade baseado nos vizinhos. (Etapa 2)"""
        # Esta é a função "faminta" que será paralelizada.
        # Ela apenas LÊ da grade e calcula deltas, não modifica nada.
        
        temp_delta = 0.0
        vx_delta = 0.0
        vy_delta = 0.0
        
        # AQUI ESTÁ A MÁGICA: Em vez de 10.000, checamos talvez 10-20.
        neighbors = grid.query_neighbors(self) 
        
        for p2 in neighbors:
            dx = self.x - p2.x
            dy = self.y - p2.y
            dist_sq = dx*dx + dy*dy
            
            if dist_sq < INTERACTION_RADIUS_SQ and dist_sq > 0.01:
                dist = math.sqrt(dist_sq)
                norm_dist = 1 - dist / INTERACTION_RADIUS
                
                # Condução térmica
                temp_diff = self.temp - p2.temp
                heat_transfer = temp_diff * THERMAL_CONDUCTIVITY * norm_dist
                temp_delta -= heat_transfer # Eu perco calor
                
                # Repulsão
                repulsion = 0.05 * norm_dist
                vx_delta += (dx / dist) * repulsion
                vy_delta += (dy / dist) * repulsion
        
        return (temp_delta, vx_delta, vy_delta)

    def apply_deltas(self, deltas):
        """Aplica as mudanças calculadas na Etapa 2. (Etapa 3)"""
        self.temp += deltas[0]
        self.vx += deltas[1]
        self.vy += deltas[2]
        
    def draw(self, canvas):
        cv2.circle(canvas, (int(self.x), int(self.y)), PARTICLE_RADIUS, self.get_color(), -1)

# ==============================================================================
# 2. FUNÇÕES DE "HELPER" PARA O MULTIPROCESSAMENTO
# ==============================================================================
# O Multiprocessing não pode chamar métodos de classe diretamente (dá erro de "pickle").
# Criamos essas funções "wrapper" que rodam no nível superior.

def _helper_update_movement(particle, fan_direction):
    """Wrapper para a Etapa 1: Movimento."""
    particle.update_movement(fan_direction)
    return particle

def _helper_calculate_interactions(particle, grid):
    """Wrapper para a Etapa 2: Interações (a parte pesada)."""
    return particle.calculate_interaction_deltas(grid)

# ==============================================================================
# 3. CLASSE PRINCIPAL DA SIMULAÇÃO
# ==============================================================================
class Simulation:
    def __init__(self, fan_direction, title):
        self.fan_direction = fan_direction
        self.title = title
        self.particles = []
        self.avg_temp = ROOM_TEMP_INITIAL
        self.grid = Grid() # Cada simulação tem sua própria grade
        self.init_particles()

    def init_particles(self):
        self.particles = []
        for _ in range(int(TARGET_PARTICLE_COUNT * 0.6)):
            x = FAN_POSITION + 20 + random.random() * (WIDTH - FAN_POSITION - 30)
            y = random.random() * HEIGHT
            temp = ROOM_TEMP_INITIAL + (random.random() - 0.5) * 4
            self.particles.append(Particle(x, y, temp))
        
        for _ in range(int(TARGET_PARTICLE_COUNT * 0.3)):
            x = random.random() * FAN_POSITION
            y = random.random() * HEIGHT
            temp = OUTSIDE_TEMP + (random.random() - 0.5) * 2
            self.particles.append(Particle(x, y, temp))

    def spawn_particles(self):
        for _ in range(SPAWN_RATE):
            if len(self.particles) >= TARGET_PARTICLE_COUNT * 1.2: break
            y = random.random() * HEIGHT
            temp = OUTSIDE_TEMP + (random.random() - 0.5) * 2
            x = 5 if self.fan_direction == 'in' else WIDTH - 5
            self.particles.append(Particle(x, y, temp))

    def update(self, pool):
        """O novo loop de update, agora usando o 'pool' de processamento."""
        
        # 0. Gerar novas partículas (serial, rápido)
        self.spawn_particles()

        # ==================
        # ETAPA 1: MOVIMENTO (Paralelo)
        # Atualiza física, ventilador e paredes para todas as partículas.
        # functools.partial "pré-configura" a função com os argumentos que não mudam.
        move_func = functools.partial(_helper_update_movement, fan_direction=self.fan_direction)
        
        # pool.map distribui a lista self.particles entre todos os núcleos
        # e aplica a função 'move_func' em cada um.
        processed_particles = pool.map(move_func, self.particles)
        
        # Filtrar partículas que morreram (saíram da tela)
        self.particles = [p for p in processed_particles if p.is_alive]
        
        # ==================
        # ETAPA 2: CONSTRUIR GRADE (Serial, rápido)
        # Limpa a grade e insere as novas posições das partículas.
        self.grid.clear()
        for p in self.particles:
            self.grid.insert(p)
            
        # ==================
        # ETAPA 3: CALCULAR INTERAÇÕES (Paralelo - A PARTE MAIS PESADA)
        # Cada núcleo calcula as deltas de calor/repulsão para um subconjunto de partículas.
        interact_func = functools.partial(_helper_calculate_interactions, grid=self.grid)
        
        # Isso usa TODOS OS SEUS NÚCLEOS para o trabalho pesado.
        all_deltas = pool.map(interact_func, self.particles)
        
        # ==================
        # ETAPA 4: APLICAR DELTAS (Serial, rápido)
        # Aplica as mudanças calculadas de volta nas partículas.
        for i, particle in enumerate(self.particles):
            particle.apply_deltas(all_deltas[i])

        # ------------------
        # Calcular temperatura (serial, rápido)
        temp_sum = 0
        count = 0
        sensor_x, sensor_y, sensor_r_sq = WIDTH - 100, 50, 60**2
        
        for p in self.particles:
            if p.x > FAN_POSITION + 20: # Otimização: só checar partículas "dentro"
                if (p.x - sensor_x)**2 + (p.y - sensor_y)**2 < sensor_r_sq:
                    temp_sum += p.temp
                    count += 1
        
        if count > 5: # Mínimo de 5 partículas para evitar ruído
            self.avg_temp = temp_sum / count

    def draw(self, canvas):
        # Esta função é a mesma de antes, roda em um só núcleo (desenho)
        canvas[:] = BG_COLOR
        cv2.circle(canvas, (WIDTH - 100, 50), 60, SENSOR_COLOR, 2)
        cv2.line(canvas, (FAN_POSITION, 0), (FAN_POSITION, HEIGHT), FAN_COLOR, 8)
        
        # Otimização: desenhar apenas X partículas aleatórias se forem muitas
        max_draw = 10000
        if len(self.particles) > max_draw:
            draw_list = random.sample(self.particles, max_draw)
        else:
            draw_list = self.particles
            
        for p in draw_list:
            p.draw(canvas)
            
        cv2.putText(canvas, self.title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
        temp_str = f"{self.avg_temp:.1f}C"
        (w, h), _ = cv2.getTextSize(temp_str, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(canvas, (WIDTH - w - 30, 20), (WIDTH - 10, 60 + h), (0,0,0,0.5), -1)
        cv2.putText(canvas, temp_str, (WIDTH - w - 20, 50 + h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2)
        cv2.putText(canvas, f"Particulas: {len(self.particles)}", (10, HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

def main():
    # Detecta quantos núcleos você tem e cria um "pool" de trabalhadores
    num_cores = os.cpu_count()
    print(f"Iniciando pool de multiprocessamento com {num_cores} nucleos...")
    
    # O 'with' garante que o pool seja fechado corretamente no final
    with multiprocessing.Pool(processes=num_cores) as pool:
        
        sim1 = Simulation(fan_direction='in', title="Ventilador -> DENTRO")
        sim2 = Simulation(fan_direction='out', title="Ventilador <- FORA")
        
        canvas1 = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        canvas2 = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        
        last_time = time.time()
        frame_count = 0
        fps = 0

        print(f"Iniciando simulação de ALTA PERFORMANCE com {TARGET_PARTICLE_COUNT * 2} partículas.")
        print("Pressione 'q' na janela da simulação para sair.")

        while True:
            # ==================
            # LÓGICA (CPU - Todos os Núcleos)
            # ==================
            sim1.update(pool) # Passa o pool para a simulação
            sim2.update(pool)
            
            # ==================
            # DESENHO (CPU - 1 Núcleo)
            # ==================
            sim1.draw(canvas1)
            sim2.draw(canvas2)
            
            combined_canvas = np.hstack((canvas1, canvas2))
            
            frame_count += 1
            now = time.time()
            if now - last_time > 1.0:
                fps = frame_count / (now - last_time)
                last_time = now
                frame_count = 0
                
            cv2.putText(combined_canvas, f"FPS: {fps:.1f}", (WIDTH - 70, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            diff = sim1.avg_temp - sim2.avg_temp
            if abs(diff) < 0.5: winner_text = "EMPATE TECNICO"
            elif diff < 0: winner_text = "PUXAR AR FRIO VENCE"
            else: winner_text = "EXPELIR AR QUENTE VENCE"
            cv2.putText(combined_canvas, winner_text, (WIDTH - 150, HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Simulacao de Alta Performance (Multicore)", combined_canvas)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    cv2.destroyAllWindows()
    print("Simulação encerrada.")

if __name__ == "__main__":
    # Essencial para o multiprocessing funcionar corretamente no Windows/macOS
    main()