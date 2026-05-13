# Framework de Robótica con IA (CoppeliaSim + PyRep + RL)

## 1. Resumen
El framework permite la simulación y el control por aprendizaje por refuerzo (RL) de robots multifuncionales dentro de CoppeliaSim. Los usuarios pueden:
- Importar o crear escenas 3D (CoppeliaSim `.ttt`)
- Configurar tareas y entrenamiento mediante YAML
- Entrenar políticas de RL para un solo agente o multi-agente (Ray RLlib)
- Extender con entornos personalizados sin modificar el código base

Los entornos de ejemplo principales incluyen la navegación de un solo robot y un escenario de alcance de objetivo cooperativo de dos robots implementado en [`DynamicTwoPhaseNavEnv`](examples/multirobot/envs/multirobot_env.py).

## 2. Características Principales
- Wrapper de PyRep para ciclos rápidos de reset / step
- Abstracción de robots (sensores, ruedas, pinzas) mediante `RobotController`
- RL multi-agente (escena compartida, lógica de coordinación)
- Ejecutor de entrenamiento universal [`run_rl_task.py`](run_rl_task.py)
- Recompensas basadas en potencial y moldeadas (progreso de distancia, alineación de rumbo, alineación de velocidad, manejo de inactividad/bloqueo)
- Aislamiento del espacio de trabajo del usuario (`user_workspace/`) para adiciones personalizadas
- Registro de checkpoints y métricas (`logs/`, `checkpoints/`)

## 3. Estructura del Repositorio (Condensada)
```
.
├── run_rl_task.py                 # Ejecutor universal de entrenamiento RL
├── examples/                      # Entornos de ejemplo, escenas y YAMLs de tareas
│   └── multirobot/envs/multirobot_env.py
├── src/
│   ├── core/                      # Simulación central / control
│   ├── robots/                    # Metadatos / ayudantes del modelo del robot
│   ├── scripts/                   # Scripts de simulación (legado)
│   └── utils/                     # Herramientas de exportación / utilidad
├── user_workspace/                # Entornos/escenas/tareas personalizadas del usuario (sandbox seguro)
├── docs/                          # Guías (básicas + avanzadas)
├── logs/                          # Registros de entrenamiento y ejecución
└── checkpoints/                   # Checkpoints de RL guardados
```

## 4. Inicio Rápido
```bash
git clone <repository-url> ai_robotics_framework
cd ai_robotics_framework
pip install -r requirements.txt
# (Opcional) export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 run_rl_task.py --task_yaml examples/multirobot/tasks/multirobot.yaml --iterations 300
```

## 5. Prerrequisitos
- Linux (o WSL2) – Requisito de PyRep
- CoppeliaSim instalado (GUI opcional si se usa `--headless`)
- Se recomienda Python 3.9–3.11
- GPU opcional (las políticas son pequeñas por defecto)

## 6. Ejecutor de Entrenamiento Universal
Archivo: [`run_rl_task.py`](run_rl_task.py)  
Admite dos formas de especificar el entorno en el YAML de la tarea:

Clase punteada (preferida):
```yaml
env_class: examples.multirobot.envs.multirobot_env.DynamicTwoPhaseNavEnv
```

Respaldo de archivo:
```yaml
env_file: user_workspace/custom_envs/my_nav_env.py
env_class_name: MyCustomNavEnv
```

Ejecutar:
```bash
python3 run_rl_task.py --task_yaml examples/multirobot/tasks/multirobot.yaml --iterations 500
python3 run_rl_task.py --task_yaml examples/navigation/tasks/navigation_easy.yaml --iterations 300
```
Flags comunes:
- `--headless`
- `--checkpoint_path <dir_o_archivo>`
- `--train_batch_size 16000`
- `--override_env_file ...` (sobrescribir YAML)

## 7. Esenciales del YAML de Tarea
Fragmento de ejemplo (multi-robot):
```yaml
scene_file: examples/multirobot/scenes/multi_nav.ttt
env_class: examples.multirobot.envs.multirobot_env.DynamicTwoPhaseNavEnv
robots_setup:
  - { type: "AstiPioneerHybrid", name: "AstiPioneer1" }
  - { type: "AstiPioneerHybrid", name: "AstiPioneer2" }
max_episode_steps: 500
success_dist: 0.40
reward_weights:
  progress: 5.0
  completion: 25.0
dynamic_obstacle:
  enabled: true
  name: MidObstacle
```

Navegación de un solo robot (mínimo):
```yaml
scene_file: examples/navigation/scenes/navigation_easy.ttt
env_class: src.core.navigation_env.NavigationEnv
robots_setup:
  - { type: "AstiPioneerHybrid", name: "AstiPioneer1" }
max_episode_steps: 360
success_dist: 0.2
```

## 8. Moldeado de Recompensas (Entorno de Dos Robots)
[`DynamicTwoPhaseNavEnv`](examples/multirobot/envs/multirobot_env.py) combina:
- Progreso de distancia (delta recortado; `w_progress` configurable)
- Moldeado basado en potencial Φ = −α·dist con descuento γ (invariante a la política)
- Penalizaciones adaptativas por inactividad + bloqueo (umbrales escalados por distancia)
- Bono de alineación de rumbo
- Alineación de velocidad: velocidad de avance proyectada en la dirección del objetivo (`vel_align_scale`)
- Penalización por giro para titubeos rotacionales
- Moldeado de rampa cerca del objetivo: bono continuo dentro de `near_target_radius_mult*success_dist`
- Penalización por colisión
Parámetros configurables a través de `reward_weights` y claves de moldeado (por ejemplo, `shaping_gamma`, `idle_warmup_steps`, `spin_penalty_scale`).

## 9. Lógica de Fase Multi-Agente
Fase 1: Cada robot recibe uno de 4 objetivos (distintos al azar).  
Fase 2: El primero en terminar elige el más cercano de los dos restantes; el otro robot recibe el último objetivo.  
El episodio termina cuando ambos han completado dos objetivos o se alcanza el límite de pasos.

## 10. Flujo de Trabajo del Espacio de Trabajo del Usuario
Los usuarios crean o extienden sin editar el núcleo:
```
user_workspace/
  custom_envs/        # Clases de entorno Gym / MultiAgent personalizadas
  custom_tasks/       # Definiciones de tareas YAML
  custom_scenes/      # Escenas .ttt importadas
```
Referenciarlos mediante:
```yaml
scene_file: user_workspace/custom_scenes/arena_variant.ttt
env_file: user_workspace/custom_envs/my_nav_env.py
env_class_name: MyNavEnv
```

## 11. Creación de un Entorno Personalizado (Plantilla)
Vea la plantilla completa en la guía avanzada o copie la forma mínima:
```python
class MyNavEnv(gym.Env):
    def __init__(self, env_config): ...
    def reset(self, *, seed=None, options=None): ...
    def step(self, action): ...
```

## 12. Registros y Checkpoints
- Registros: `logs/robot_controller.log`, `logs/rl_training.log`, registros de ejecución personalizados creados a través de `setup_logger` en [`src/core/logger.py`](src/core/logger.py)
- CSV de Métricas: `logs/training_metrics.csv`
- Checkpoints: `checkpoints/<fecha_o_id_de_ejecución>/`
Restaurar:
```bash
python3 run_rl_task.py --task_yaml ... --checkpoint_path checkpoints/run_123
```

## 13. Extensión de Robots
Agregue modelos (.ttm) bajo `src/robots/` (o carpeta personalizada) y las definiciones correspondientes en su ayudante de fábrica de definiciones de robots (vea las definiciones existentes referenciadas en la construcción del controlador).

## 14. Exportación / Despliegue (Futuro)
Planificado:
- Exportación ONNX para políticas entrenadas
- Hook de programación de currículo en el ejecutor
- Paquete de aleatorización de dominio (ruido de sensores, fricción)

## 15. Solución de Problemas (Condensada)
| Problema | Solución |
|----------|----------|
| Fallo en la importación de PyRep | Confirmar que se está ejecutando en Linux y que la ruta de CoppeliaSim está configurada si es necesario |
| ImportError de env_class | Agregar el archivo `__init__.py` faltante o usar `env_file` / `env_class_name` |
| Desajuste de tipo de objeto (Dummy vs Shape) | Usar el resolutor unificado en el `RobotController.set_object_pose` actualizado |
| Progresión de recompensa baja al principio | Aumentar `w_progress`, reducir `idle_warmup_steps`, verificar el escalado de velocidad |

Más detalles: vea la guía avanzada.

## 16. Contribuir
1. Fork y rama
2. Agregar pruebas/ejemplos si corresponde
3. Mantener la documentación actualizada (`README.md` + guía avanzada)
4. Enviar PR

## 17. Licencia
MIT (vea `LICENSE`)

---
Para notas técnicas más profundas (matemáticas de moldeado de potencial, migración de empaquetado, trampas), consulte `docs/ADVANCED_GUIDE.md`.
