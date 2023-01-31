

#include <QTableView>
#include <QHeaderView>
#include <QStandardItemModel>
#include <QPushButton>
#include <QRadioButton>
#include <QGroupBox>
#include <QLabel>
#include <QPainter>
#include <QChart>
#include <QLineEdit>
#include <QToolButton>
#include <QLineSeries>
#include <QScatterSeries>
#include <QtCharts>
#include <QRandomGenerator>
#include <QTimer>
#include <QDebug>

#include <QVBoxLayout>

#include <dialog/validator/OutlinedQLineEdit.h>
#include <dialog/validator/SimpleDoubleValidator.h>

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <sys/sysinfo.h>


#include <vector>
#include <math.h>
#include <cmath>
#include <iostream>

#include <ProjectManagerDatabaseUpdate.h>


ProjectmanagerDatabaseUpdate::ProjectmanagerDatabaseUpdate(QWidget* parent)
{

	setWindowTitle("Project manager database update");
	QVBoxLayout * mainLayout00 = new QVBoxLayout(this);


	QHBoxLayout *layout1 = new QHBoxLayout;
	QLabel *qlabel_session = new QLabel("Session");
	qlineedit_session = new QLineEdit;
	qlineedit_session->setReadOnly(true);
	QPushButton *qpb_sessionload = new QPushButton("Load");
	layout1->addWidget(qlabel_session);
	layout1->addWidget(qlineedit_session);
	layout1->addWidget(qpb_sessionload);

	QPushButton *qpb_update = new QPushButton("Update");

	mainLayout00->addLayout(layout1);
	mainLayout00->addWidget(qpb_update);

	connect(qpb_sessionload, SIGNAL(clicked()), this, SLOT(trt_session_load()));
	connect(qpb_update, SIGNAL(clicked()), this, SLOT(trt_update()));

	load_last_session();
}


ProjectmanagerDatabaseUpdate::~ProjectmanagerDatabaseUpdate()
{

}

void ProjectmanagerDatabaseUpdate::load_last_session()
{
	const QString lastPath = settings.value(LAST_SESSION_PATH_IN_SETTINGS, "").toString();
	if (lastPath.isEmpty() || !QFileInfo(lastPath).exists())
		return;

	session_filename = lastPath;
	session_name = lastPath;
	qlineedit_session->setText(session_name);
}

void ProjectmanagerDatabaseUpdate::trt_session_load()
{
	    QSettings settings;
	    GlobalConfig& config = GlobalConfig::getConfig();
	    const QString lastPath = settings.value(LAST_SESSION_PATH_IN_SETTINGS, config.sessionPath()).toString();

	    const QString filePath = QFileDialog::getOpenFileName(this,
	                                                          tr("Load Session"),
	                                                          lastPath,
	                                                          QLatin1String("*.json"));
	    if (filePath.isEmpty())
	        return;
	    session_filename = filePath;
	    session_name = filePath;
	    qlineedit_session->setText(session_name);

	    const QFileInfo fi(filePath);
	    settings.setValue(LAST_SESSION_PATH_IN_SETTINGS, fi.absoluteFilePath());
	   // m_selectorWidget->load_session(filePath);
}

void ProjectmanagerDatabaseUpdate::trt_update()
{
	QFile file(session_filename);
	if (!file.open(QIODevice::ReadOnly))
	{
		qDebug() << "GeotimeProjectManagerWidget : cannot load session, file not readable";
		return;
	}
	QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
	if (!doc.isObject())
	{
		qDebug() << "GeotimeProjectManagerWidget : cannot load session, root is not a json object";
		return;
	}

	QJsonObject rootObj = doc.object();
	QString projectType = rootObj["projectType"].toString("None");
	QString project = rootObj["project"].toString("None");
	QString survey = rootObj["survey"].toString("None");
	qDebug() << projectType << project << survey;



/*

	QJsonArray jsonArray = mainObject["definition"].toArray();

	QJsonObject object = jsonArray[1].toObject();
	QStringList ql = object.keys();


	QString txt2 = object["Name"].toString("None");


	QString txt = mainObject.value("tap").toString("None");
	// QString txt2 = mainObject.value("definition").toString("None");



	QJsonObject a1 = mainObject["images"].toObject();
	QJsonArray a2 = a1["Properties"].toArray();
	QString txt3 = a2[0].toString("None");
	*/





}





