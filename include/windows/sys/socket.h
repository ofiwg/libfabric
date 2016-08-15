
#ifndef _LIBFABRIC_OSD_SYS_SOCKET_H_
#define _LIBFABRIC_OSD_SYS_SOCKET_H_

/* on Windows we have to follow strong sequence of includes:
   winsock2.h & ws2tcpip.h should be included prior to
   windows.h */

#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>

#endif /* _LIBFABRIC_OSD_SYS_SOCKET_H_ */

