#include <stdio.h>
#include "tvm_wrapper.h"

int main() {
    printf("Hello\n");
    printf("Init: ");
    TVMWrap_Init();
    printf("done!\n");
    printf("Run: ");
    TVMWrap_Run();
    printf("done!\n");
    return 0;

}
