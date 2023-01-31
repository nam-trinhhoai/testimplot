
#include <iostream>

#include <Xt.h>

#include <checkDataSizeMatch.h>


bool checkSeismicsSizeMatch(const QString& cube1, const QString& cube2)
{
	if ( cube1.isNull() || cube1.isEmpty() || cube2.isNull() || cube2.isEmpty()) {
		return false;
	}
	inri::Xt xt1(cube1.toStdString().c_str());
	if (!xt1.is_valid()) {
		std::cerr << "xt cube is not valid (" << cube1.toStdString() << ")" << std::endl;
		return false;
	}
	inri::Xt xt2(cube2.toStdString().c_str());
	if (!xt2.is_valid()) {
		std::cerr << "xt cube is not valid (" << cube2.toStdString() << ")" << std::endl;
		return false;
	}

	bool match = xt1.nRecords()==xt2.nRecords() && xt1.nSamples()==xt2.nSamples() && xt1.nSlices()==xt2.nSlices() &&
			xt1.stepRecords()==xt2.stepRecords() && xt1.stepSamples()==xt2.stepSamples() &&
			xt1.stepSlices()==xt2.stepSlices() && xt1.nRecords()==xt2.nRecords() && xt1.nSamples()==xt2.nSamples() &&
			xt1.nSlices()==xt2.nSlices();
	return match;
}

