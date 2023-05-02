# Create carla sif file if it does not exist
CARLA_VERSION=0.9.13
CARLA_SIF = "carla-$(CARLA_VERSION).sif"
CARLA_DEF = "./carla_server.def"

CARLA_SERVER_PORT?=2000
CARLA_SERVER_GPU_DEVICE?=0


# .PHONY: all clean
#
# all: 
# 	trap 'kill -- -$$' SIGINT SIGTERM EXIT

build-carla:
	if [ ! -f "$(CARLA_SIF)" ]; then \
		apptainer build $(CARLA_SIF) $(CARLA_DEF); \
	fi

run-carla: build-carla
	apptainer exec --nv $(CARLA_SIF) /home/carla/CarlaUE4.sh --world-port=$(CARLA_SERVER_PORT) -RenderOffScreen \
	-ini:[/Script/Engine.RendererSettings]:r.GraphicsAdapter=$(CARLA_SERVER_GPU_DEVICE) && break; \
