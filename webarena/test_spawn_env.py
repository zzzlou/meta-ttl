import multiprocessing as mp
import os

def worker():
    import sys
    return os.environ.get('TEST_SPAWN_VAR', 'NOT_FOUND')

if __name__ == '__main__':
    mp.set_start_method('spawn')
    os.environ['TEST_SPAWN_VAR'] = 'spawn_success'
    
    with mp.Pool(1) as pool:
        result = pool.apply(worker)
        print("Result from worker:", result)
