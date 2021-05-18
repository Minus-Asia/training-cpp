#include "./shared_lib/Hello_Name.h"
#include <iostream>

int main(int argc, char *argv[])
{
    char name[100];
    hello HiName;
    std::cin.getline(name,100);
    HiName.showName(name);
    return 0;
}