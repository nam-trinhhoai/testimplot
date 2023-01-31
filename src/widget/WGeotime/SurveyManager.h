
#ifndef __SURVEYMANAGER__
#define __SURVEYMANAGER__

#include <QWidget>
#include <QString>
#include <QLineEdit>
#include <QLabel>
#include <QComboBox>
#include <QCheckBox>
#include <QListWidget>
#include <QDir>
#include <QLineEdit>
#include <QTabWidget>
#include <QGroupBox>
#include <QTableWidget>
#include <QPushButton>
#include <QVBoxLayout>


#include <vector>
#include <math.h>
#include <utility>

#include <ProjectManagerNames.h>
#include <SeismicManager.h>
#include <HorizonManager.h>
#include <RgbRawManager.h>

class SurveyManager : public QWidget{
	Q_OBJECT
public:
	SurveyManager(QWidget* parent = 0);
	virtual ~SurveyManager();
	QListWidget *listwidgetSurvey;
	QString getName();
	QString getPath();
	void setProjectSelectedPath(int projectType, QString path, QString name);
	void setSeismicManager(SeismicManager *seismicManager);
	void setHorizonManager(HorizonManager *horizonManager);
	void setRgbRawManager(RgbRawManager *rgbRawManager);
	bool setForceName(QString name);

signals:
	void surveyChanged();

private:
	const QString surveyRootSubDir = "DATA/3D/";
	SeismicManager *m_seismicManager;
	HorizonManager *m_horizonManager;
	RgbRawManager *m_rgbRawManager;

	QLineEdit *lineeditSurveySearch;
	QString m_projectFullPath, m_projectName;
	int m_idxProjectType;
	ProjectManagerNames m_names;
	std::vector<QString> getDirList(QString path);
	void updateNames();
	void updateDisplay();
	std::pair<QString, QString> getSurveyPath();


public slots:
	void trt_SearchChange(QString str);
	void trt_surveylistClick(QListWidgetItem*item);





};



#endif
