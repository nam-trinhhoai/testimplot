

#ifndef __AVIVIEWWIDGET__
#define __AVIVIEWWIDGET__



#include <QThread>
#include <QWidget>
#include <QString>
#include <QLineEdit>
#include <QLabel>
#include <QCheckBox>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QComboBox>
#include <QGroupBox>
#include <QDialog>
#include <QString>

#include <string>
#include <vector>
#include <string>
#include <math.h>

#include <fileSelectWidget.h>


class AviViewWidget : public QWidget{
	Q_OBJECT

private:

public:
	AviViewWidget(ProjectManagerWidget *selectorWidget, QWidget* parent = 0);
	virtual ~AviViewWidget();

private:
	FileSelectWidget *m_aviFileSelectWidget = nullptr;
	ProjectManagerWidget *m_selectorWidget = nullptr;

	private slots:
	void trt_run();
};





#endif
