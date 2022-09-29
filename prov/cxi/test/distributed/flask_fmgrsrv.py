#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
# Copyright (c) 2022 Hewlett Packard Enterprise Development LP


import argparse
import sys
import os
from flask import Flask, jsonify, abort, make_response, request

# Database of multicast addresses
addresses = {}
max_index = 0
max_mcast = 100

app = Flask('__name__')

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route('/')
def index():
    return "Simulated REST API for multicast creation\n" \
           "  GET    /mcasts     - provide list of all multicast addresses\n" \
           "  POST   /mcasts     - create a new multicast address\n" \
           "  GET    /mcast/<id> - list an existing multicast address\n" \
           "  DELETE /mcast/<id> - delete an existing multicast address\n"

@app.route('/mcasts', methods=['GET'])
def get_mcasts():
    addrlist = [{idx: addresses[idx]} for idx in addresses]
    return jsonify(addrlist)

@app.route('/mcasts', methods=['POST'])
def create_mcasts():
    global max_index, max_mcast

    mcast_id = max_index
    addresses[mcast_id] = {
        'mcast': max_mcast,
        'hwroot': 0
    }
    max_index += 1
    max_mcast += 3
    return jsonify({mcast_id : addresses[mcast_id]}), 201

@app.route('/mcast/<int:mcast_id>', methods=['GET'])
def get_mcast(mcast_id):
    if mcast_id not in addresses:
        abort(404)
    return jsonify({mcast_id : addresses[mcast_id]}), 201

@app.route('/mcast/<int:mcast_id>', methods=['DELETE'])
def delete_mcast(mcast_id):
    if mcast_id not in addresses:
        abort(404)
    del(addresses[mcast_id])
    return jsonify({'result': True})

if __name__ == '__main__':
    # Get host IP address
    hostip=os.environ.get('HOSTIP')
    if hostip is None:
        hostip='0.0.0.0'

    # Allow host IP to be overridden
    parser = argparse.ArgumentParser(
        description='''
        Simulated Fabric Manager REST service.
        ''')
    parser.add_argument('--ipaddr', type=str, default=hostip,
                        help='server IP address')
    args = parser.parse_args()
    parser.parse_args(sys.argv[1:], namespace=args)

    # Start the application
    app.run(debug=True, host=args.ipaddr)
