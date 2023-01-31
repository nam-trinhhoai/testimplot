#ifndef SRC_SLICER_GRAPHICSREP_ITOOLTIPPROVIDER_H
#define SRC_SLICER_GRAPHICSREP_ITOOLTIPPROVIDER_H

#include <QString>

class IToolTipProvider {
public:
	virtual QString generateToolTipInfo() const = 0;
};

#endif

