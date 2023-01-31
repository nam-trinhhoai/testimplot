#ifndef __ISOHORIZON__
#define __ISOHORIZON__

#include <QObject>
#include <QPointer>
#include <QVector2D>
#include <QList>
#include <QColor>
#include <vector>

#include <fixedrgblayersfromdatasetandcube.h>
#include <fixedlayerimplisohorizonfromdatasetandcube.h>
#include "itreewidgetitemdecoratorprovider.h"

class SeismicSurvey;
#include "idata.h"
#include "viewutils.h"
// #include "RgtLayerProcessUtil.h"

class IconTreeWidgetItemDecorator;
class IsoHorizonGraphicRepFactory;
class IsoHorizonAttribut;
class StackSynchronizer;

class IsoHorizon: public IData, public ITreeWidgetItemDecoratorProvider {
Q_OBJECT

public:
class Attribut
{
public:
	FixedRGBLayersFromDatasetAndCube *pFixedRGBLayersFromDatasetAndCube = nullptr;
	FixedLayerImplIsoHorizonFromDatasetAndCube *pFixedLayerImplIsoHorizonFromDatasetAndCube = nullptr;

	IData* data();
};

public:
	IsoHorizon(WorkingSetManager * workingSet, SeismicSurvey *survey, const QString &path, const QString &name, QObject *parent = 0);
	virtual ~IsoHorizon();

	//IData
	virtual IGraphicRepFactory *graphicRepFactory();
	QUuid dataID() const override;
	QString name() const override{return m_name;}

	// QList<WellPick*> getWellPickFromWell(WellBore* bore);
	// const QList<WellPick*>& wellPicks() const;
	QColor color() const;
	void setColor(const QColor& color);
	// QList<RgtSeed> getProjectedPicksOnDataset(Seismic3DAbstractDataset* dataset, int channel, SampleUnit sampleUnit);

	// ITreeWidgetItemDecoratorProvider
	virtual ITreeWidgetItemDecorator* getTreeWidgetItemDecorator() override;

	IData* getIsochronData();

	/*
public slots:
	void addWellPick(WellPick* pick);
	void removeWellPick(WellPick* pick);

signals:
	void wellPickAdded(WellPick* pick);
	void wellPickRemoved(WellPick* pick);
	*/
	// std::vector<FreeHorizonAttribut*> m_attribut;


	// std::vector<FixedRGBLayersFromDatasetAndCube*> m_attribut;
	std::vector<Attribut> m_attribut;

signals:
	void colorChanged(QColor color);

	// only used by item decorator
	void iconChanged(QIcon icon);

private slots:
	void updateIcon(QColor color);

private:
	QString m_path = "";
	QString m_name;
	QUuid m_uuid;
	QObject *m_parent = nullptr;
	WorkingSetManager * m_workingSet = nullptr;
	SeismicSurvey *m_survey = nullptr;
	QPointer<FixedLayerImplIsoHorizonFromDatasetAndCube> m_isoData;

	IsoHorizonGraphicRepFactory* m_repFactory;
	// QList<WellPick*> m_wellPicks;
	IconTreeWidgetItemDecorator* m_decorator;
	SampleUnit m_sampleUnit;
	QColor m_color;
	void horizonAttributCreate();

	StackSynchronizer* m_synchronizer = nullptr;
};

#endif
