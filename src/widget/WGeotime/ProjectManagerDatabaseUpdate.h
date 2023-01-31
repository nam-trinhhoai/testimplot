
#ifndef __PROJECTMANAGERDATABASEUPDATE__
#define __PROJECTMANAGERDATABASEUPDATE__

#include <QThread>
#include <QWidget>
#include <QString>
#include <QLineEdit>
#include <QLabel>
#include <QCheckBox>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QComboBox>
#include <QGroupBox>
#include <QDialog>


#include <vector>
#include <math.h>

#include "GeotimeProjectManagerWidget.h"





class ProjectmanagerDatabaseUpdate : public QWidget{
	Q_OBJECT
public:
	ProjectmanagerDatabaseUpdate(QWidget* parent = 0);
	virtual ~ProjectmanagerDatabaseUpdate();


private:
	void load_last_session();

	QLineEdit *qlineedit_session;
	GeotimeProjectManagerWidget *m_selectorWidget;
	QString session_filename;
	QString session_name;



private slots:
	void trt_session_load();
	void trt_update();

};









#endif
