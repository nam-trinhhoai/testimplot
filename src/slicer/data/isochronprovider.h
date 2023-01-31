#ifndef IsoChronProvider_H
#define IsoChronProvider_H
#include "cpuimagepaletteholder.h"
#include <QPointF>

class CPUImagePaletteHolder;

class  IsoSurfaceBuffer
{
public:
	std::shared_ptr<CPUImagePaletteHolder> buffer=nullptr;
	//CPUImagePaletteHolder* buffer=nullptr;
	float originSample=0.0f;
	float stepSample=1.0f;

	IsoSurfaceBuffer()
	{

	}

	IsoSurfaceBuffer(const IsoSurfaceBuffer& other)
	{
		buffer=other.buffer;
		originSample=other.originSample;
		stepSample= other.stepSample;
	}

	IsoSurfaceBuffer& operator=(const IsoSurfaceBuffer& other)
	{
		this->originSample = other.originSample;
		this->stepSample = other.stepSample;
		this->buffer = other.buffer;
		return *this;
	}

	float getAltitude(QPointF pt)
	{
		if( buffer != nullptr)
		{

			double heightValue = 0.0;
			int i = 0;
			int j = 0;
			bool res = buffer->value(pt.x(),pt.y(),i,j,heightValue);
			if( !res ) return 0.0f;

			float newHeightValue =  originSample + stepSample *heightValue;
			return newHeightValue;
		}

		qDebug()<<"-->IsoSurfaceBuffer , le buffer est null";
		return 0.0f;
	}

	bool isValid()
	{
		return (buffer != nullptr);
	}

};


class IsoChronProvider
{

public:

	virtual IsoSurfaceBuffer getIsoBuffer()=0;
};


#endif
