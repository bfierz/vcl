# Triage Labels

The skills speak in terms of five canonical triage roles. This file maps those roles to the actual label strings used in this repo's issue tracker.

| Label in mattpocock/skills | Label in our tracker | Meaning                                                                                                    |
| -------------------------- | -------------------- | ---------------------------------------------------------------------------------------------------------- |
| `needs-triage`             | `needs-triage`       | Maintainer needs to evaluate this issue                                                                    |
| `needs-info`               | `needs-info`         | Waiting on reporter for more information                                                                   |
| `ready-for-agent`          | `ready-unattended`   | Fully specified, and an agent can build and verify it alone (Linux dev container or CI)                    |
| `ready-for-human`          | `ready-supervised`   | An agent can build it, but a human must be in the loop: Windows or GPU verification, or a judgement call |
| `wontfix`                  | `wontfix`            | Will not be actioned                                                                                       |

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), use the corresponding label string from this table. Use `ready-supervised` wherever a skill says `ready-for-human`.

Choosing between the two ready labels: if verifying the acceptance criteria needs anything the Linux dev container and CI can't provide (a Windows host, MSVC, a D3D12 device, a specific GPU or driver, macOS), the ticket is `ready-supervised`.

## Area labels

Area labels sit beside the triage role. Apply every one that fits; omit them when an issue is platform- or backend-neutral.

| Label              | Scope                                     |
| ------------------ | ----------------------------------------- |
| `platform:windows` | Windows builds (MSVC, clang-cl, MinGW)    |
| `platform:linux`   | Linux builds (gcc, clang, ARM cross)      |
| `platform:macos`   | macOS builds (apple-clang)                |
| `platform:web`     | Emscripten / browser builds               |
| `backend:opengl`   | OpenGL graphics backend (incl. EGL, GLEW) |
| `backend:vulkan`   | Vulkan graphics backend                   |
| `backend:d3d12`    | Direct3D 12 graphics backend              |
| `backend:webgpu`   | WebGPU graphics backend (Dawn)            |
| `backend:cuda`     | CUDA compute libraries                    |
| `backend:opencl`   | OpenCL compute libraries                  |
