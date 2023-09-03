#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define ALIVE 1
#define DEAD 0

// Function prototypes
void print_board(int *board, int rows, int cols, int large_rank_rows, int large_ranks);
void initialize_board(int *board, int rows, int cols);
void update_board(int *board, int rows, int cols);
void exchange_ghost_rows(int *board, int rows, int cols, int rank, int size);

int main(int argc, char **argv) {
    int rows, cols;     // Dimensions of board
    int i, j;     // Loop counters
    int rank, size; // Rank of current process and total number of processes
    int iters;
    MPI_Status status;  // Status of MPI communication

    if (argc != 4) {
        printf("Usage: %s <rows> <cols> <iterations>\n", argv[0]);
        return 1;
    }

    rows = atoi(argv[1]);
    cols = atoi(argv[2]);
    iters = atoi(argv[3]);

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double init_time, final_time, time_elapsed;

    if (rank == 0) {
        init_time = MPI_Wtime();
        printf("Initial time: %f\n", init_time);
    }

    // Calculate the number of rows owned by each process
    int rows_per_rank, start_row;
    int large_ranks = rows % size;
    if (rank < large_ranks) {
        rows_per_rank = (rows / size) + 1;
        start_row = rank * rows_per_rank;
    } else {
        rows_per_rank = rows / size;
        start_row = large_ranks * (rows_per_rank + 1) + 
                    (rank - large_ranks) * rows_per_rank;

    }
    int total_rank_rows = rows_per_rank + 2;

    int end_row = start_row + rows_per_rank;

    // Initialize board for each rank
    int board[total_rank_rows][cols];
    initialize_board(board, rows_per_rank, cols);


    MPI_Barrier(MPI_COMM_WORLD);
    // Main game loop
    for (int i = 0; i < iters; i++) {
        MPI_Barrier(MPI_COMM_WORLD);

        // Exchange ghost rows with neighboring processes
        exchange_ghost_rows(board, rows_per_rank, cols, rank, size);

        // Update the board
        update_board(board, rows_per_rank, cols);

        // Gather the board from all processes to rank 0
        if (rank == 0) {
            int full_board[rows][cols];
            for (int r = 0; r < rows_per_rank; r++) {
                for (int c = 0; c < cols; c++) {
                    full_board[r][c] = board[r + 1][c];
                }
            }
            for (int recv = 1; recv < size; recv++) {
                if (recv < large_ranks) {
                    MPI_Recv(&full_board[recv*rows_per_rank][0], rows_per_rank*cols, MPI_INT, recv, 0, MPI_COMM_WORLD, &status);
                } else {
                    int recv_rows = large_ranks > 0 ? rows_per_rank - 1 : rows_per_rank;
                    int index = large_ranks * rows_per_rank + (recv - large_ranks) * recv_rows;
                    MPI_Recv(&full_board[index][0], recv_rows*cols, MPI_INT, recv, 0, MPI_COMM_WORLD, &status);
                }
            }
            print_board(full_board, rows, cols, rows_per_rank, large_ranks);
        } else {
            MPI_Send(&board[1][0], rows_per_rank*cols, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }
    if (rank == 0) {
        final_time = MPI_Wtime();
        time_elapsed = final_time - init_time;
        printf("Final time: %f\n", final_time);
        printf("Time elapsed: %f\n", time_elapsed);
    }
    MPI_Finalize();
    return 0;
}

void initialize_board(int *board, int rows, int cols) {
    int i, j;
    for (i = 0; i < rows + 2; i++) {
        for (j = 0; j < cols; j++) {
            *(board + i*cols + j) = (i == 0 || i == rows + 1) ? DEAD : rand() % 2;
        }
    }
}

void update_board(int *board, int rows, int cols) {
    int i, j, m, n, count;
    int new_board[rows][cols];
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            count = 0;
            for (m = -1; m <= 1; m++) {
                for (n = -1; n <= 1; n++) {
                    if ((m == 0 && n == 0) || j+n < 0 || j+n >= cols) {
                        continue;
                    }
                    if (*(board + (i + m + 1)*cols + j + n)) {
                        count++;
                    }
                }
            }
            if (*(board + (i + 1)*cols + j)) {
                if (count < 2 || count > 3) {
                    new_board[i][j] = DEAD;
                } else {
                    new_board[i][j] = ALIVE;
                }
            } else {
                if (count == 3) {
                    new_board[i][j] = ALIVE;
                } else {
                    new_board[i][j] = DEAD;
                }
            }
        }
    }
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            *(board + (i + 1)*cols + j) = new_board[i][j];
        }
    }
}

void exchange_ghost_rows(int *board, int rows, int cols, int rank, int size) {
    MPI_Status status;
    if (rank != size - 1) {
        MPI_Send((board + rows*cols), cols, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
        MPI_Recv((board + (rows + 1)*cols), cols, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, &status);
    }
    if (rank != 0) {
        MPI_Send(board + cols, cols, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
        MPI_Recv(board, cols, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &status);
    }
}

void print_board(int *board, int rows, int cols, int large_rank_rows, int large_ranks) {
    int i, j;
    int large_ranks_section = large_rank_rows * large_ranks;
    _Bool large_exists = large_ranks > 0;
    for (i = 0; i < rows; i++) {
        int color;
        if (!large_exists || i < large_ranks_section) {
            color = i / large_rank_rows;
        } else {
            color = large_ranks + (i - large_ranks_section) / (large_rank_rows - 1);
        }
        for (j = 0; j < cols; j++) {
            if (*(board + i*cols + j)) {
                printf("\033[1;3%dm1 ", color + 1);
            } else {
                printf("\033[0m0 ");
            }
        }
        printf("\033[0m\n");
    }
    printf("\n");
}