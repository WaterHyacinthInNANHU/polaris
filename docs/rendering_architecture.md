# Rendering Architecture

This document explains the hybrid rendering architecture in Polaris DROID environments, which combines physics simulation with photorealistic Gaussian splat rendering.

## Overview

Polaris uses a **dual representation system**:

1. **USD Meshes** - For physics simulation (collision, rigid body dynamics)
2. **Gaussian Splats** - For photorealistic scene rendering

These two systems run in parallel and are synchronized through transform updates.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      Polaris Environment                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────┐     ┌──────────────────────────────┐ │
│  │   Isaac Sim/Lab      │     │    Gaussian Splat Renderer   │ │
│  │   (Physics Engine)   │     │    (gsplat/nerfstudio)       │ │
│  ├──────────────────────┤     ├──────────────────────────────┤ │
│  │ • USD Stage          │     │ • 3DGS Point Cloud           │ │
│  │ • Rigid Body Physics │────▶│ • Photorealistic Rendering   │ │
│  │ • Collision Detection│     │ • Camera Transforms          │ │
│  │ • Robot Simulation   │     │ • Scene Background           │ │
│  └──────────────────────┘     └──────────────────────────────┘ │
│           │                              │                      │
│           ▼                              ▼                      │
│  ┌──────────────────────┐     ┌──────────────────────────────┐ │
│  │   Sim Render Output  │     │   Splat Render Output        │ │
│  │   (Robot + Objects)  │     │   (Photorealistic Scene)     │ │
│  └──────────────────────┘     └──────────────────────────────┘ │
│           │                              │                      │
│           └──────────────┬───────────────┘                      │
│                          ▼                                      │
│                 ┌──────────────────┐                            │
│                 │   Compositing    │                            │
│                 │   (Mask-based)   │                            │
│                 └──────────────────┘                            │
│                          │                                      │
│                          ▼                                      │
│                 ┌──────────────────┐                            │
│                 │   Final Image    │                            │
│                 │   (RGB Output)   │                            │
│                 └──────────────────┘                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. USD Meshes (Physics Layer)

The USD stage contains mesh representations used for:
- **Collision detection** - Objects interact physically
- **Rigid body dynamics** - Objects respond to forces, gravity
- **Robot kinematics** - Joint positions and velocities
- **Object tracking** - Position and orientation queries

Key USD APIs used:
- `PhysicsRigidBodyAPI` - Enables rigid body simulation
- `PhysxRigidBodyAPI` - PhysX-specific physics properties
- `PhysicsCollisionAPI` - Collision geometry
- `PhysxConvexDecompositionCollisionAPI` - Convex decomposition for complex shapes

### 2. Gaussian Splats (Rendering Layer)

The Gaussian splat representation provides:
- **Photorealistic rendering** - Real-world appearance captured from images
- **View-dependent effects** - Reflections, specular highlights
- **Scene background** - Kitchen environment, lighting conditions
- **Fast rendering** - Real-time differentiable rendering

The splat data is stored as a point cloud with:
- Position (xyz)
- Color (RGB spherical harmonics)
- Opacity (alpha)
- Covariance (3D Gaussian shape)

### 3. Transform Synchronization

Object poses from the physics simulation are synchronized to the splat renderer:

```python
def transform_sim_to_splat(self, transform_dict: dict):
    """Apply simulation transforms to splat scene objects.

    Args:
        transform_dict: {object_name: (position, quaternion)}
    """
    for obj_name, (pos, quat) in transform_dict.items():
        if obj_name in self.splat_scene.objects:
            self.splat_scene.set_object_transform(obj_name, pos, quat)
```

This ensures that when objects move in the physics simulation, their visual appearance in the splat render updates accordingly.

### 4. Compositing Pipeline

The final image is created by compositing the robot from simulation with the splat-rendered scene:

```python
def custom_render(self):
    # 1. Render photorealistic scene from Gaussian splats
    splat_img = self.render_splat(camera_pose)

    # 2. Render robot from Isaac Sim
    sim_img, mask = self.get_robot_from_sim()

    # 3. Composite: robot pixels from sim, background from splat
    final_img = np.where(mask, sim_img, splat_img)

    return final_img
```

The mask identifies robot pixels in the simulation render, allowing them to be overlaid on the photorealistic splat background.

## Background Rendering (DomeLight)

The environment background (sky/ambient lighting) comes from the USD stage's lighting setup:

```usda
def Xform "Environment"
{
    def DistantLight "defaultLight" (
        prepend apiSchemas = ["ShapingAPI"]
    )
    {
        float inputs:angle = 1
        float inputs:intensity = 3000
        float inputs:shaping:cone:angle = 180
        quatd xformOp:orient = (0.653, 0.271, 0.271, 0.653)
    }
}
```

- **DistantLight** - Simulates sunlight (parallel rays)
- **DomeLight** - Environment map for sky/background (when used)

The splat renderer captures these lighting conditions from the original scene capture, providing consistent illumination.

## Data Flow

1. **Physics Step**: Isaac Sim advances simulation, updates object poses
2. **Pose Query**: Environment queries current object transforms from USD stage
3. **Transform Sync**: Object poses sent to splat renderer
4. **Splat Render**: Gaussian splat renderer produces photorealistic image
5. **Sim Render**: Isaac Sim renders robot with segmentation mask
6. **Composite**: Final image combines robot (sim) with scene (splat)
7. **Output**: RGB observation returned to policy

## File Locations

| Component | Location |
|-----------|----------|
| Splat rendering | `src/polaris/environments/manager_based_rl_splat_environment.py` |
| USD scenes | `PolaRiS-Hub/<env_name>/scene.usda` |
| Splat data | `PolaRiS-Hub/<env_name>/splat/` (3DGS checkpoint) |
| Environment config | `src/polaris/environments/droid_cfg.py` |

## Benefits of Hybrid Approach

1. **Photorealism** - Gaussian splats capture real-world appearance
2. **Physical Accuracy** - USD meshes provide accurate collision/dynamics
3. **Sim-to-Real Transfer** - Photorealistic rendering reduces domain gap
4. **Flexibility** - Can modify physics without re-capturing scene
5. **Performance** - Efficient rendering of complex scenes

## Limitations

- **Object Appearance** - Manipulated objects use mesh rendering, not splats
- **Lighting Changes** - Splat lighting is baked from capture conditions
- **New Objects** - Adding objects requires mesh assets (can't add to splat)
