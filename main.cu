///////////////////////////////////////////////////////////////////////////
/// PROGRAMACIÓN EN CUDA C/C++
/// Práctica:	BASICO 4 : Arrays Multidimensionales
/// Autor:		Gustavo Gutierrez Martin
/// Fecha:		Octubre 2022
///////////////////////////////////////////////////////////////////////////

/// dependencias ///
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "cuda_runtime.h"

/// constantes ///
#define MB (1<<20) /// MiB = 2^20
#define ROWS 6
#define COLUMNS 21

/// muestra por consola que no se ha encontrado un dispositivo CUDA
int getErrorDevice();
/// muestra los datos de los dispositivos CUDA encontrados
int getDataDevice(int deviceCount);
/// numero de CUDA cores
int getCudaCores(cudaDeviceProp deviceProperties);
/// muestra por pantalla las propiedades del dispositivo CUDA
int getDeviceProperties(int deviceId, int cudaCores, cudaDeviceProp cudaProperties);
/// inicializa el array del host
int loadHostData(int *hst_vector1, int *hst_vector2);
/// transferimos los datos del host al device
int dataTransferToDevice(int *hst_vector1, int *dev_vector1);
/// realiza la suma de los arrays en el device
__global__ void transfer(int *dev_vector1, int *dev_vector2);
/// transfiere los datos del device al host
int dataTransferToHost(int *hst_vector2, int *dev_vector2);
/// muestra por pantalla los datos del host
int printData(int *hst_vector1, int *hst_vector2);
/// función que muestra por pantalla la salida del programa
int getAppOutput();

int main() {
    int deviceCount;
    dim3 blocks(1);
    dim3 threads(ROWS, COLUMNS);
    int *hst_vector1, *dev_vector1;
    int *hst_vector2, *dev_vector2;

    /// buscando dispositivos
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        /// mostramos el error si no se encuentra un dispositivo
        return getErrorDevice();
    } else {
        /// mostramos los datos de los dispositivos CUDA encontrados
        getDataDevice(deviceCount);
    }
    /// reserva del espacio de memoria en el host
    hst_vector1 = (int*)malloc( ROWS * COLUMNS * sizeof(int));
    hst_vector2 = (int*)malloc( ROWS * COLUMNS * sizeof(int));
    /// reserva del espacio de memoria en el device
    cudaMalloc( (void**)&dev_vector1, ROWS * COLUMNS * sizeof(int) );
    cudaMalloc( (void**)&dev_vector2, ROWS * COLUMNS * sizeof(int) );
    /// cargamos los datos del host
    loadHostData(hst_vector1, hst_vector2);
    /// transferimos los datos del host al device
    dataTransferToDevice(hst_vector1, dev_vector1);
    /// mostramos los datos con los que llamamos al device
    printf("Lanzamiento de: %d bloque y %d hilos \n", 1, threads.x * threads.y);
    printf("> Eje X: %d \n", threads.x);
    printf("> Eje Y: %d \n", threads.y);
    printf("***************************************************\n");
    /// sumamos los items
    transfer<<< blocks, threads >>>(dev_vector1, dev_vector2);
    /// transferimos los datos del device al host
    dataTransferToHost(hst_vector2,dev_vector2);
    /// muestra por pantalla los datos del host
    printData(hst_vector1,hst_vector2);
    /// función que muestra por pantalla la salida del programa
    getAppOutput();
    /// liberamos los recursos del device
    cudaFree(dev_vector1);
    cudaFree(dev_vector2);
    return 0;
}

int getErrorDevice() {
    printf("¡No se ha encontrado un dispositivo CUDA!\n");
    printf("<pulsa [INTRO] para finalizar>");
    getchar();
    return 1;
}

int getDataDevice(int deviceCount) {
    printf("Se han encontrado %d dispositivos CUDA:\n", deviceCount);
    for (int deviceID = 0; deviceID < deviceCount; deviceID++) {
        ///obtenemos las propiedades del dispositivo CUDA
        cudaDeviceProp deviceProp{};
        cudaGetDeviceProperties(&deviceProp, deviceID);
        getDeviceProperties(deviceID, getCudaCores(deviceProp), deviceProp);
    }
    return 0;
}

int getCudaCores(cudaDeviceProp deviceProperties) {
    int cudaCores = 0;
    int major = deviceProperties.major;
    if (major == 1) {
        /// TESLA
        cudaCores = 8;
    } else if (major == 2) {
        /// FERMI
        if (deviceProperties.minor == 0) {
            cudaCores = 32;
        } else {
            cudaCores = 48;
        }
    } else if (major == 3) {
        /// KEPLER
        cudaCores = 192;
    } else if (major == 5) {
        /// MAXWELL
        cudaCores = 128;
    } else if (major == 6 || major == 7 || major == 8) {
        /// PASCAL, VOLTA (7.0), TURING (7.5), AMPERE
        cudaCores = 64;
    } else {
        /// ARQUITECTURA DESCONOCIDA
        cudaCores = 0;
        printf("¡Dispositivo desconocido!\n");
    }
    return cudaCores;
}

int getDeviceProperties(int deviceId, int cudaCores, cudaDeviceProp cudaProperties) {
    int SM = cudaProperties.multiProcessorCount;
    printf("***************************************************\n");
    printf("DEVICE %d: %s\n", deviceId, cudaProperties.name);
    printf("***************************************************\n");
    printf("- Capacidad de Computo            \t: %d.%d\n", cudaProperties.major, cudaProperties.minor);
    printf("- No. de MultiProcesadores        \t: %d \n", SM);
    printf("- No. de CUDA Cores (%dx%d)       \t: %d \n", cudaCores, SM, cudaCores * SM);
    printf("- Memoria Global (total)          \t: %zu MiB\n", cudaProperties.totalGlobalMem / MB);
    printf("- No. maximo de Hilos (por bloque)\t: %d\n", cudaProperties.maxThreadsPerBlock);
    printf("***************************************************\n");
    return 0;
}

int loadHostData(int *hst_vector1, int *hst_vector2) {
    srand ( (int)time(nullptr) );
    for (int i=0; i < ROWS * COLUMNS; i++)  {
        /// inicializamos hst_vector1 con numeros aleatorios entre 0 y 1
        hst_vector1[i] = (int) rand() % 10;
    }
    return 0;
}

int dataTransferToDevice(int *hst_vector1, int *dev_vector1) {
    /// transfiere datos de hst_A a dev_A
    cudaMemcpy(dev_vector1,hst_vector1, ROWS * COLUMNS * sizeof(int),cudaMemcpyHostToDevice);
    return 0;
}

__global__ void transfer(int *dev_vector1, int *dev_vector2) {
    /// identificador del hilo
    unsigned int threadX = threadIdx.y;
    unsigned int threadY = threadIdx.x;
    /// calculamos el ID  hilo
    unsigned int myID = threadY + threadX * blockDim.x;
    /// calculamos la fila donde se encuentra la posicion
    int row = (int) myID / COLUMNS;
    /// calculamos si la posicion
    if ((myID - (row * COLUMNS)) % 2 == 0 ) {
        dev_vector2[myID] = dev_vector1[myID];
    } else {
        dev_vector2[myID] = 0;
    }

}

int dataTransferToHost(int *hst_vector2, int *dev_vector2) {
    /// transfiere datos de dev_vector2 a hst_vector2
    cudaMemcpy(hst_vector2, dev_vector2, ROWS * COLUMNS * sizeof(int), cudaMemcpyDeviceToHost);
    return 0;
}

int printData(int *hst_vector1, int *hst_vector2) {
    printf("MATRIZ ORIGINAL:\n");
    for (int i = 0; i < ROWS; i++)  {
        for (int j = 0; j < COLUMNS; j++) {
            printf("%d ", hst_vector1[j + i * COLUMNS]);
        }
        printf("\n");
    }
    printf("\n");
    printf("MATRIZ FINAL:\n");
    for (int i = 0; i < ROWS; i++)  {
        for (int j = 0; j < COLUMNS; j++) {
            printf("%d ", hst_vector2[j + i * COLUMNS]);
        }
        printf("\n");
    }
    printf("\n");
    return 0;
}

int getAppOutput() {
    /// salida del programa
    time_t fecha;
    time(&fecha);
    printf("***************************************************\n");
    printf("Programa ejecutado el: %s", ctime(&fecha));
    printf("***************************************************\n");
    /// capturamos un INTRO para que no se cierre la consola de MSVS
    printf("<pulsa [INTRO] para finalizar>");
    getchar();
    return 0;
}
