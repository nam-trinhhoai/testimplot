#ifndef StratiSlice_H
#define StratiSlice_H

#include <QObject>
#include "idata.h"

class Seismic3DAbstractDataset;
class IGraphicRepFactory;
class AmplitudeStratiSliceAttribute;
class RGBStratiSliceAttribute;
class FrequencyStratiSliceAttribute;

class StratiSlice: public IData {
Q_OBJECT
public:
	StratiSlice(WorkingSetManager *workingSet,
			Seismic3DAbstractDataset *seismic, Seismic3DAbstractDataset *rgt,
			QObject *parent = 0);
	virtual ~StratiSlice();

	RGBStratiSliceAttribute *rgbAttribute() const{return m_rgbAttribute; }
	AmplitudeStratiSliceAttribute *amplitudeAttribute()const{return m_amplitudeAttribute; }
	FrequencyStratiSliceAttribute *frequencyAttribute()const{return m_frequencyAttribute; }

	unsigned int width() const;
	unsigned int height() const;
	unsigned int depth() const;

	Seismic3DAbstractDataset* seismic() const {
		return m_seismic;
	}

	Seismic3DAbstractDataset* rgt() const {
		return m_rgt;
	}

	//IData
	QUuid dataID() const override;
	QString name() const override;

	QVector2D rgtMinMax();

	virtual IGraphicRepFactory* graphicRepFactory()override{return m_repFactory;}
private:
	Seismic3DAbstractDataset *m_seismic;
	Seismic3DAbstractDataset *m_rgt;

	RGBStratiSliceAttribute *m_rgbAttribute;
	AmplitudeStratiSliceAttribute *m_amplitudeAttribute;
	FrequencyStratiSliceAttribute *m_frequencyAttribute;

	IGraphicRepFactory *m_repFactory;

};
Q_DECLARE_METATYPE(StratiSlice*)
#endif
