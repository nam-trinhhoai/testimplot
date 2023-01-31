#ifndef SRC_WIDGET_IMPORTSISMAGEHORIZONWITHWOKRINGSETWIDGET_H
#define SRC_WIDGET_IMPORTSISMAGEHORIZONWITHWOKRINGSETWIDGET_H

#include <QWidget>

class IData;
class ImportSismageHorizonWidget;
class Seismic3DAbstractDataset;
class SeismicSurvey;
class WorkingSetManager;

class QComboBox;

class ImportSismageHorizonWithWorkingSetWidget : public QWidget {
	Q_OBJECT
public:
	struct SurveyConnections {
		QMetaObject::Connection addConnection;
		QMetaObject::Connection removeConnection;
	};

	ImportSismageHorizonWithWorkingSetWidget(WorkingSetManager* manager, QWidget* parent=0);
	~ImportSismageHorizonWithWorkingSetWidget();

signals:
	void horizonExtracted(QString horizonName, QString datasetPath);
	void importFinished();

protected:
	virtual void showEvent(QShowEvent* event) override;

private slots:
	void currentDatasetChanged(int index);
	void datasetAdded(Seismic3DAbstractDataset* dataset);
	void datasetRemoved(Seismic3DAbstractDataset* dataset);
	void potentialSurveyAdded(IData* data);
	void potentialSurveyRemoved(IData* data);
	void relayHorizonExtracted(QString horizonName, QString datasetPath);
	void relayImportFinished();

	void updateHorizonsInTree();

	void updateComboBox(bool force=false);



private:
	ImportSismageHorizonWidget* m_importerWidget;
	QComboBox* m_datasetComboBox;
	Seismic3DAbstractDataset* m_currentDataset;
	WorkingSetManager* m_manager;

	std::map<SeismicSurvey*, SurveyConnections> m_connections;
};

#endif // SRC_WIDGET_IMPORTSISMAGEHORIZONWITHWOKRINGSETWIDGET_H
