import threading

import numpy as np

from utils import announce


def test_threading(sequential = False):
    def thread_function(container, index):
        values = np.random.randint(10000, size=1000)
        values = np.outer(values, values)
        result = np.average(np.square(values))
        container[index] = result


    amount_of_threads = 120
    result_list = [None for _ in range(amount_of_threads)]
    threads = [threading.Thread(target=thread_function, args=(result_list, i))
               for i in range(amount_of_threads)]


    def execute_multithreaded():
        print("Staring the threads")
        for thread in threads:
            thread.start()

        print("Waiting for the threads")
        for thread in threads:
            thread.join()

        print("Threads have finished")

    def execute_sequential():
        print("Executing sequentially")
        for thread in threads:
            thread.start()
            thread.join()


        print("Sequential Threads have finished")

    if sequential:
        execute_sequential()
    else:
        execute_multithreaded()


    print("The final output is", result_list)



with announce("Multi"):
    test_threading(False)

with announce("Sequential"):
    test_threading(True)






