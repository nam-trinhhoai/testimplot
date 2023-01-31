#ifndef Dataset3DSliceRep_H
#define Dataset3DSliceRep_H

#include <QObject>
#include <QVector2D>
#include <QUuid>
#include <QMap>
#include <QPair>
#include <QMutex>
#include "abstractgraphicrep.h"
#include "idatacontrolerholder.h"
#include "sliceutils.h"
#include "idatacontrolerprovider.h"
#include "imouseimagedataprovider.h"
#include "affinetransformation.h"
#include "isampledependantrep.h"

class CUDAImagePaletteHolder;
class QGLLineItem;
class Dataset3DPropPanel;
class Dataset3DSliceLayer;
class DataControler;
class Seismic3DAbstractDataset;

class Dataset3DSliceRep:
		public AbstractGraphicRep, public ISampleDependantRep {
Q_OBJECT
public:
	Dataset3DSliceRep(Seismic3DAbstractDataset *data, CUDAImagePaletteHolder *slice,
			const QPair<QVector2D,AffineTransformation> & sliceRangeAndTransfo, SliceDirection dir =
					SliceDirection::Inline, AbstractInnerView *parent = 0);
	virtual ~Dataset3DSliceRep();

	CUDAImagePaletteHolder* image() {
		return m_image;
	}

	SliceDirection direction() const {
		return m_dir;
	}

	int currentSliceWorldPosition() const;
	int currentSliceIJPosition() const;

	int channel() const;
	void setChannel(int channel);

	//AbstractGraphicRep
	QWidget* propertyPanel() override;
	Graphic3DLayer * layer3D(QWindow * parent,Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera) override;

	IData* data() const override;


	QPair<QVector2D,AffineTransformation> sliceRangeAndTransfo() const;
	bool setSampleUnit(SampleUnit sampleUnit) override;
	QList<SampleUnit> getAvailableSampleUnits() const override;
	QString getSampleUnitErrorMessage(SampleUnit sampleUnit) const override;
	virtual TypeRep getTypeGraphicRep() override;
public slots:
	void setSliceWorldPosition(int pos);
	void setSliceIJPosition(int pos);
	void setSlicePosition(int posWorld, int posIJ);
	void delete3DRep();
signals:
	void sliceWordPositionChanged(int pos);
	void sliceIJPositionChanged(int pos);
	void channelChanged(int channel);
	void deletedRep(AbstractGraphicRep *rep);// MZR 15072021
private slots:
	void dataChanged();
private:
	QString generateName();
	void loadSlice(unsigned int z);
	void setSlicePositionInternal(int posWorld, int posIJ);
private:

	unsigned int m_currentSlice;
	CUDAImagePaletteHolder *m_image;
	SliceDirection m_dir;
	Dataset3DPropPanel *m_propPanel;

	Dataset3DSliceLayer *m_layer;

	//Range min/max of the data available
	QPair<QVector2D,AffineTransformation>  m_sliceRangeAndTransfo;

	Seismic3DAbstractDataset *m_data;
	int m_channel;

	mutable QMutex m_writeMutex;
	mutable QMutex m_nextMutex;
	bool m_isNextDefined = false;
	int m_nextWorldPos;
	int m_nextIJPos;
};

#endif
