### Just a thing for me rn, I'll come back
from numba import cuda


def get_memory(id_):
    with cuda.gpus[id_]:
        free, total = cuda.current_context().get_memory_info()
    return free, total

if cuda.is_available():
    # Internet ways
    devices = cuda.gpus
    print(devices[0])
    try_device = devices[0].name
    device = cuda.get_current_device().name
    print(f'Name: {device}, {try_device}')

    print('-'*50)
    # Slide way
    cuda.detect()
    device = cuda.select_device(0)
    print(device)

    id_ = device.id
    name_ = device.name
    print(id_)
    print(name_)

    # Next thing
    mp_count = device.MULTIPROCESSOR_COUNT 
    print(mp_count)
    my_cc = device.compute_capability
    print(my_cc)


    cc_cores_per_SM_dict = {
    (2,0) : 32,
    (2,1) : 48,
    (3,0) : 192,
    (3,5) : 192,
    (3,7) : 192,
    (5,0) : 128,
    (5,2) : 128,
    (6,0) : 64,
    (6,1) : 128,
    (7,0) : 64,
    (7,5) : 64,
    (8,0) : 64,
    (8,6) : 128,
    (8,9) : 128,
    (9,0) : 128,
    (10,0) : 128,
    (12,0) : 128
    }
    cores_per_sm = cc_cores_per_SM_dict.get(my_cc)
    total_cores = cores_per_sm*mp_count

    print(total_cores)

    free, total = get_memory(id_)
    print(free)
    print(total)

