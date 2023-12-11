#include <iostream>
#include <vector>

//void seq_mat_mul_sdot(int N, std::vector<float>& A, std::vector<float>& B, std::vector<float>& C);

void seq_mat_mul_sdot(int N, std::vector<float>& A, std::vector<float>& B, std::vector<float>& C)
{
    int i, j, k;
    float tmp;

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            tmp = 0.0f;
            std::cout << "i="<< i << "j=" << j << "tmp=" << tmp<<"\n";
            for (k = 0; k < N; k++) {
                /* C(i,j) = sum(over k) A(i,k) * B(k,j) */
                
                tmp += A[i*N+k] * B[k*N+j];
                std::cout << tmp << " ";
                
            }
            C[i*N+j] = tmp;
            std::cout << "\n" << C[i*N+j]  << "\n";
        }
    }
}
int main() {
    // Define the matrix size
    const int N = 3;

    // Create matrices A, B, and C
    std::vector<float> A = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0};
    std::vector<float> B = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0};
    std::vector<float> C(N * N, 0.0);

    // Perform matrix multiplication
    seq_mat_mul_sdot(N, A, B, C);

    // Display the matrices
    std::cout << "Matrix A:" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << A[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nMatrix B:" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << B[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nResultant Matrix C:" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
