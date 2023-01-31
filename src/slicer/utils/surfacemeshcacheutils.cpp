#include "surfacemeshcacheutils.h"

QByteArray byteArrayFromRawData(const char* data, long size) {
	QByteArray output;
	output.resize(size);
	memcpy(output.data(), data, size);
	return output;
}
