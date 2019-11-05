#include <pthread.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#define NUM_THREADS 6
int counter = 0;
int thread_count;
pthread_mutex_t barrier_mutex;

void *Thread_work(void *rank)
{
    long my_rank = (long)rank;
    printf("Before barrier: %ld\n", my_rank);
    /*Barrier*/
    pthread_mutex_lock(&barrier_mutex);
    counter++;
    pthread_mutex_unlock(&barrier_mutex);
    while (counter < NUM_THREADS)
        ;
    printf("After barrier: %ld\n", my_rank);
}

int main()
{

    long thread;
    pthread_t *thread_handles = (pthread_t *)malloc(NUM_THREADS * sizeof(pthread_t));
    
    /* Start threads */
    for (thread = 0; thread < NUM_THREADS; thread++)
        pthread_create(&thread_handles[thread], NULL, Thread_work, (void *)thread);

    /* Wait for threads to complete */
    for (thread = 0; thread < NUM_THREADS; thread++)
        pthread_join(thread_handles[thread], NULL);
}
