#!/bin/bash

docker run  --volume `pwd`:/root/units --workdir /root/units -it --rm -p 127.0.0.1:8888:8888 ldssa/lisbon.ai:3
