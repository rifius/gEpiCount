#!/bin/sh
#
revision=`git describe`
versionfile="src/version.cpp"
tmpfile=`mktemp`

cat <<ENDVERSION > ${tmpfile}
#include	<string>
#include	"../inc/gEpiCount.h"

const string progVersion = "${revision}";

ENDVERSION

[[ -f ${versionfile} ]] || touch ${versionfile}
diff -B -b -w ${versionfile} ${tmpfile} || cat ${tmpfile} > ${versionfile}

rm -f ${tmpfile}


