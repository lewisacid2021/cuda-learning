#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <stdlib.h>

__global__ void infiniteKernel(int *data)
{
    long long i = 0;
    while (1) {
        data[i % 10] += 1;
        i++;
        if (i % 100000000 == 0)
            printf("Kernel running, i=%lld\n", i);
    }
}

int main()
{
    int pipefd[2];
    if (pipe(pipefd) != 0) {
        perror("pipe failed");
        return 1;
    }

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork failed");
        return 1;
    }

    if (pid == 0) {
        // 子进程
        close(pipefd[0]);

        int *d_data;
        cudaMalloc(&d_data, 10 * sizeof(int));
        cudaMemset(d_data, 0, 10 * sizeof(int));

        printf("[Child] Launching infinite kernel...\n");
        infiniteKernel<<<1, 1>>>(d_data);
        cudaDeviceSynchronize(); // 会阻塞，实际上 kernel 无限循环

        // 通知父进程 kernel 执行完成（这里永远不会执行到）
        write(pipefd[1], "done", 4);
        close(pipefd[1]);
        return 0;
    } else {
        // 父进程
        close(pipefd[1]);

        printf("[Parent] Sleeping 2s before killing child...\n");
        sleep(2);

        printf("[Parent] Killing child process to free GPU context...\n");
        kill(pid, SIGKILL);

        int status;
        waitpid(pid, &status, 0);

        printf("[Parent] Child exited. GPU context should be released.\n");

        // 测试再次使用 GPU
        int *d_test;
        cudaError_t err = cudaMalloc(&d_test, 10 * sizeof(int));
        printf("[Parent] cudaMalloc after child exit: %s\n", cudaGetErrorString(err));
        cudaFree(d_test);

        return 0;
    }
}
