/*
 * Copyright (C) 2025 Cornelis Networks.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef HFISVC_CLIENT_EXPORT_H
#define HFISVC_CLIENT_EXPORT_H

#ifndef HFISVC_CLIENT_EXPORT
#define HFISVC_CLIENT_EXPORT __attribute__((visibility("default")))
#endif

#ifndef HFISVC_CLIENT_NO_EXPORT
#define HFISVC_CLIENT_NO_EXPORT __attribute__((visibility("hidden")))
#endif

#ifndef HFISVC_CLIENT_DEPRECATED
#define HFISVC_CLIENT_DEPRECATED __attribute__((__deprecated__))
#endif

#ifndef HFISVC_CLIENT_DEPRECATED_EXPORT
#define HFISVC_CLIENT_DEPRECATED_EXPORT HFISVC_CLIENT_EXPORT HFISVC_CLIENT_DEPRECATED
#endif

#ifndef HFISVC_CLIENT_DEPRECATED_NO_EXPORT
#define HFISVC_CLIENT_DEPRECATED_NO_EXPORT HFISVC_CLIENT_NO_EXPORT HFISVC_CLIENT_DEPRECATED
#endif

#endif /* HFISVC_CLIENT_EXPORT_H */
