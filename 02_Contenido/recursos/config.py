from pathlib import Path

# Semilla Global para reproducibilidad
RANDOM_SEED = 42

# Colores corporativos / tema (opcional)
COLORS = ["#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600"]

# Configuración de Gráficos
FIG_SIZE = (10, 6)

# Paths relativos (asumiendo que se importa desde un notebook en una subcarpeta)
# Esto ayuda a localizar recursos compartidos si fuera necesario
ROOT_DIR = Path(__file__).parent.parent
