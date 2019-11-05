#include <pthread.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#define NUM_THREADS 6
int counter = 0;
sem_t count_sem;
sem_t barrier_sem;
void *Thread_work(void *rank)
{
    long my_rank = (long)rank;
    printf("Before barrier: %ld\n", my_rank);
    /*Barrier*/
    sem_wait(&count_sem);
    if (counter == NUM_THREADS - 1)
    {
        counter = 0;
        sem_post(&count_sem);
        for (int j = 0; j < NUM_THREADS - 1; j++)
            sem_post(&barrier_sem);
    }
    else
    {
        counter++;
        sem_post(&count_sem);
        sem_wait(&barrier_sem);
    }
    printf("After barrier: %ld\n", my_rank);
}
int main()
{
    sem_init(&count_sem, 0, 1);
    sem_init(&barrier_sem, 0, 0);
    long thread;
    pthread_t *thread_handles = (pthread_t *)malloc(NUM_THREADS * sizeof(pthread_t));

    /* Start threads */
    for (thread = 0; thread < NUM_THREADS; thread++)
        pthread_create(&thread_handles[thread], NULL, Thread_work, (void *)thread);

    /* Wait for threads to complete */
    for (thread = 0; thread < NUM_THREADS; thread++)
        pthread_join(thread_handles[thread], NULL);
}
