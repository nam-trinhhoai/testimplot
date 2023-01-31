#ifndef ISampleDependantRep_H
#define ISampleDependantRep_H

#include <QString>
#include <QList>
#include "viewutils.h"

/*
 * A graphic rep who need to be aware about the slice position
 */
class ISampleDependantRep {
public:
        ISampleDependantRep() {
        }
        virtual ~ISampleDependantRep() {
        }
        virtual bool setSampleUnit(SampleUnit sampleUnit) = 0;
        virtual QList<SampleUnit> getAvailableSampleUnits() const = 0;
        virtual QString getSampleUnitErrorMessage(SampleUnit sampleUnit) const = 0;
};
#endif

