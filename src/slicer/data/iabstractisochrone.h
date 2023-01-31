#ifndef SRC_SLICER_DATA_IABSTRACTISOCHRONE_H_
#define SRC_SLICER_DATA_IABSTRACTISOCHRONE_H_

class IAbstractIsochrone {
public:
	IAbstractIsochrone() {}
	virtual ~IAbstractIsochrone() {}

	virtual int getNumTraces() const = 0;
	virtual int getNumProfils() const = 0;

	virtual float getValue(long i, long j, bool* ok) = 0;
	virtual float* getTab() = 0;
};

#endif
