
#include <stdio.h>
#include <unistd.h>

int main (int argc, char ** argv) {

	setbuf(stdout, NULL);
	setbuf(stderr, NULL);

	while (1) usleep(1000);

	return 0;
}




