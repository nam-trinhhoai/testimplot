#ifndef SRC_SLICER_INNERVIEW_IDEPTHVIEW_H
#define SRC_SLICER_INNERVIEW_IDEPTHVIEW_H

class MtLengthUnit;

class IDepthView {
public:
	virtual const MtLengthUnit* depthLengthUnit() const = 0;
	virtual void setDepthLengthUnit(const MtLengthUnit* depthLengthUnit) = 0;
};

#endif
