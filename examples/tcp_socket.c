#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <getopt.h>
#include <stdbool.h>

#define PORT 43192

char *serv_addr = NULL;
bool client_flag = 0;

int main(int argc, char *argv[])
{
	int socket_desc, client_sock, client_size, op;
	struct sockaddr_in server_addr, client_addr;
	char server_message[256];
	char client_message[256] = "Hello, server! I am the client you've been waiting for!";

	while ((op = getopt(argc, argv, "hs:c")) != -1) {
		switch (op) {
		case 's':
			serv_addr = optarg;
			break;
		case 'c':
			client_flag = 1;
			break;
		default:
		case '?':
		case 'h':
			printf(argv[0], "A simple tcp client-server example.\n");
			printf("-s <server's ip address>\n");
			printf("-c  => indicating this is client\n");
			return 0;
		}
	}

	if (!serv_addr) {
		printf("Need a server address\n");
		return -1;
	}

	// Clean buffers:
	memset(server_message, '\0', sizeof(server_message));

	// Create socket:
	socket_desc = socket(AF_INET, SOCK_STREAM, 0);

	if (socket_desc < 0) {
		printf("Error while creating socket\n");
		return -1;
	}
	printf("Socket created successfully\n");

	// Set port and IP:
	server_addr.sin_family = AF_INET;
	server_addr.sin_port = htons(PORT);
	server_addr.sin_addr.s_addr = inet_addr(serv_addr);
	if (!client_flag) {
		// Bind to the set port and IP:
		if (bind(socket_desc, (struct sockaddr*)&server_addr,
			 sizeof(server_addr)) < 0) {
			printf("Couldn't bind to the port\n");
			return -1;
		}
		printf("Binding complete\n");

		// Listen for clients:
		if (listen(socket_desc, 1) < 0) {
			printf("Error while listening\n");
			return -1;
		}
		printf("Listening for incoming connections...\n");

		// Accept an incoming connection:
		client_size = sizeof(client_addr);
		client_sock = accept(socket_desc,
				     (struct sockaddr*)&client_addr,
				     &client_size);

		if (client_sock < 0) {
			printf("Can't accept\n");
			return -1;
		}
		printf("Client connected at IP: %s and port: %i\n",
			inet_ntoa(client_addr.sin_addr),
			ntohs(client_addr.sin_port));

		// Receive client's message:
		if (recv(client_sock, client_message,
			 sizeof(client_message), 0) < 0) {
			printf("Couldn't receive\n");
			return -1;
		}
		printf("Msg from client: %s\n", client_message);

		// Respond to client:
		strcpy(server_message, "Hello! This is the server.");

		if (send(client_sock, server_message,
			 strlen(server_message), 0) < 0) {
			printf("Can't send\n");
			return -1;
		}
	}

	if (client_flag) {
		// Send connection request to server:
		if (connect(socket_desc, (struct sockaddr*)&server_addr,
			    sizeof(server_addr)) < 0) {
			printf("Unable to connect\n");
			return -1;
		}
		printf("Connected with server successfully\n");

		// Send the message to server:
		if (send(socket_desc, client_message,
			 strlen(client_message), 0) < 0) {
			printf("Unable to send message\n");
			return -1;
		}

		// Receive the server's response:
		if (recv(socket_desc, server_message,
			 sizeof(server_message), 0) < 0) {
			printf("Error while receiving server's msg\n");
			return -1;
		}
		printf("Server's response: %s\n",server_message);
	}

	close(client_sock);

	close(socket_desc);

	return 0;
}
