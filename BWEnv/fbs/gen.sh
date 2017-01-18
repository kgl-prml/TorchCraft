#!/bin/sh
exec flatc -c --gen-mutable --scoped-enums --gen-object-api messages.fbs
