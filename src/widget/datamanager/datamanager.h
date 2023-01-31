#ifndef SRC_WIDGET_DATAMANAGER_DATAMANAGER_H_
#define SRC_WIDGET_DATAMANAGER_DATAMANAGER_H_

#include <QWidget>

class ProjectManager;
class SurveyManager;
class Culturals;
class FileInformationTableWidget;
class FileDeletionTableWidget;
class TrashTableWidget;
class FileStorageControler;
class LeafContainer;

class QStackedWidget;
class QListWidget;
class QListWidgetItem;

class DataManager : public QWidget {
	Q_OBJECT
public:
	DataManager(QWidget *parent = nullptr, Qt::WindowFlags f = Qt::WindowFlags());
	~DataManager();

private slots:
	void loadSessionGui();
	void projectChanged();
	void surveyChanged();
	void clearProject();
	void clearSurvey();
	void tabListItemChanged(QListWidgetItem* current, QListWidgetItem* previous);
private:
	void loadLastSession();
	void loadSession(const QString& sessionPath);
	void addTab(QWidget* widget, const QString& tabName);

	ProjectManager* m_projectManager;
	SurveyManager* m_surveyManager;
	QStackedWidget* m_stackWidget;
	QListWidget* m_tabList;
	std::unique_ptr<Culturals> m_culturals;

	QString m_cacheProjectPath;
	QString m_cacheSurveyPath;
	FileInformationTableWidget* m_seismicTable;
	FileInformationTableWidget* m_culturalTable;
	FileInformationTableWidget* m_layerTable;
	FileInformationTableWidget* m_sismageHorizonTable;
	FileDeletionTableWidget* m_nextvisionHorizonTable;
	FileDeletionTableWidget* m_cubergt2rgbTable;
	TrashTableWidget* m_trashTable;
	FileStorageControler* m_controler;

	LeafContainer* m_nextVisionHorizonTrash;
	LeafContainer* m_cubeRgt2RgbTrash;
};

#endif
