#ifndef IMouseDataProvider_H
#define IMouseDataProvider_H

#include <QString>
#include <vector>
#include "viewutils.h"

class IMouseImageDataProvider {
public:
	IMouseImageDataProvider() {
	}
	virtual ~IMouseImageDataProvider() {
	}

	struct MouseInfo
	{
		int i;
		int j;
		bool depthValue;
		double depth;
		SampleUnit depthUnit;
		std::vector<std::string> valuesDesc;
		std::vector<double> values;
	};

	virtual bool mouseData(double x, double y,MouseInfo & info)=0;

};
#endif
