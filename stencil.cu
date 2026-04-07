#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <cuda_runtime.h>


#define TILE_SIZE 16

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA ERROR: %s:%d :%s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(-10); \
        } \
    } while (0)


/* TF am i doing, right
 *
 * Read grayscale Image
 * Copy to GPU
 * apply 2D stencil
 *      - 2D 5 point stencil 
 *      - After computation, set between 0-255
 * For Boundarys:
 *      - Do not apply stencil to borders
 *      - Replace value at borders with 0'set
 *
 * Kernel Requirements:
 *      - 2D Thread indexing
 *      - use shared memeory
 *      - Use 1-halo pixel
 *      - __syncthreads() properly
 *      - Perform Boundary checks
 *      - 1 Output pixel/Thread
 *      - 2D grid, 2D blocks
 */




static void skip_ws(FILE *fp) {

    int ch;
    while ((ch = fgetc(fp)) != EOF) {
        if (ch == '#') {

           while((ch = fgetc(fp)) != EOF && ch != '\n');

        } else if (!isspace(ch)) {

            ungetc(ch, fp);
            break;
        }
    }
}

static void get_stuff(); // idk

static int read_pgm(const char *path, uint8_t **grayscale, unsigned int *w_out, unsigned int *h_out, size_t *img_size) {
    
    FILE *input_pgm = fopen(path, "rb");

    if (input_pgm == NULL) {
        fprintf(stderr, "ERROR: Failed to open file \n");
        return -1;
    }

    //skip_ws(input_pgm);

    char magic_number[3];
    fscanf(input_pgm, "%2s", magic_number);

    if (strcmp(magic_number, "P5") != 0) {
        fprintf(stderr, "ERROR: Wrong binary format expected P5 but got %s \n", magic_number);
        fclose(input_pgm);
        return -2;
    }

    skip_ws(input_pgm);

    int w;
    int h;
    int maxval;
    
    // get the width, height, and max value from image
    if (fscanf(input_pgm, "%d", &w) != 1) {
        fprintf(stderr, "ERROR: Failed to read width from image\n");
        fclose(input_pgm);
        return -2;
    }

    skip_ws(input_pgm);

    if (fscanf(input_pgm, "%d", &h) != 1) {
        fprintf(stderr, "ERROR: Failed to read height from image \n");
        fclose(input_pgm);
        return -2;
    }  
    skip_ws(input_pgm);

    if (fscanf(input_pgm, "%d", &maxval)  != 1) { 

        fprintf(stderr, "ERROR: Failed to read data from image \n");
        fclose(input_pgm);
        return -2;
    }

    if (w <= 0 || h <= 0 || maxval <= 0) {
        fprintf(stderr, "ERROR: Unsupported PGM Header\n");
        fclose(input_pgm);
        return -3;
    }

    printf("Found w=%d h=%d maxval=%d \n", w, h, maxval);

    fgetc(input_pgm);

    size_t n_bytes = (size_t)w * (size_t)h;
    printf("n_bytes = %zu \n", n_bytes);
    uint8_t *gray_bytes = (uint8_t*)malloc(n_bytes);

    if (gray_bytes == NULL) {
        fprintf(stderr, "ERROR: Failed to allocate memory for grayscale\n");
        fclose(input_pgm);
        return -4;
    }

    size_t got_bytes = fread(gray_bytes, 1, n_bytes, input_pgm);
    if (got_bytes != n_bytes) {
        fprintf(stderr, "ERROR: Expected %zu bytes but read %zu bytes \n", n_bytes, got_bytes);
        fclose(input_pgm);
        return -2;
    }

    *grayscale = gray_bytes;
    *img_size = got_bytes;
    *w_out = (unsigned)w;
    *h_out = (unsigned)h;

    return 0;

}


__global__ void stencil_kernel(const uint8_t *gray, uint8_t *filtered, unsigned int width, unsigned int height) {

    __shared__ shared_memory[TILE_SIZE + 2][TILE_SIZE + 2];     // 1 pixel halo






}


int main(int argc, char *argv[]) {

    if (argc != 3) {
        fprintf(stderr, "ERROR: Expected 1 pos argument but %d was given. \n", argc);
        fprintf(stderr, "USAGE: ./stencil <path/to/input_ppm> <path/to/output_pgm> \n");
        return -1;
    }


    printf("Alejandro Rubio \n");
    printf("R11886363 \n");

    // ------ Inital Set Up -------
   
    const char *input = argv[1];
    const char *output = argv[2];

    printf("Input Path: %s \n", input);
    printf("Output Path: %s \n", output);
    
    int err_code = 0;
    unsigned int width, height = 0;
    size_t gray_size = 0;
    uint8_t *h_grayscale = nullptr;
    
    err_code = read_pgm(input, &h_grayscale, &width, &height, &gray_size);

    if (err_code != 0) {
        fprintf(stderr, "Program quitting ERR CODE: %d \n", err_code);
        return err_code;
    }

    printf("Image dimensions: %d X %d \n", width, height);
    printf("Image Size (bytes) %zu \n", gray_size);
    
    // ------------ GPU Set up -----------
    dim3 block(TILE_SIZE, TILE_SIZE, 1);
    dim3 grid(
            (width + TILE_SIZE - 1) / TILE_SIZE,
            (height * TILE_SIZE * 2 - 1) / (TILE_SIZE * 2)
        );

    printf("Block set up (%d:%d) \n", block.x, block.y);
    printf("Grid Set up (%d:%d) \n", grid.x, grid.y);


    uint8_t *h_output = (uint8_t*)malloc(gray_size);
    if (h_output == NULL) {
        fprintf(stderr, "ERRIR: Failed to allocate host memory for output \n");
        return -32;
    }

    // ------------ Allocate DEvice Memory ---------
    uint8_t d_grayscale, d_output = nullptr;
    CHECK_CUDA(cudaMalloc((void **)&d_grayscale, gray_size));
    CHECK_CUDA(cudaMalloc((void **)&d_output, gray_size));
    CHECK_CUDA(cudaMemcpy(d_grayscale, h_grayscale, gray_size, width, height, cudaMemcpyHostToDevice));


    printf("SUCCESSFULL\n");
    return err_code;
}
