
#ifndef __HORIZONSELECTWIDGET__
#define __HORIZONSELECTWIDGET__

class QPushButton;
class QLineEdit;
class QLabel;
class QString;
class QListWidget;
class QString;

class ProjectManagerWidget;
class WorkingSetManager;

#include <vector>

class HorizonSelectWidget : public QWidget{
	Q_OBJECT
public:
	enum TYPE { RAW, FREE };
	HorizonSelectWidget(QWidget* parent = 0);
	// HorizonSlectWidget(QString label, QString lineEditLabel,  QWidget* parent = 0);
	virtual ~HorizonSelectWidget();
	void ihmCreate();
	void setLabelText(QString txt);
	void setAddButtonLabel(QString txt);
	void setSupprButtonLabel(QString txt);
	void setProjectManager(ProjectManagerWidget *manager);
	void setListMultiSelection(bool type);
	std::vector<QString> getNames();
	std::vector<QString> getPaths();
	void setType(TYPE val) { m_type = val; }
	void setWorkingSetManager(WorkingSetManager *p) { m_workingSetManager = p; }
	void addData(QString name, QString path);
	void clearData();


private:
	// QGroupBox* qgb_seismic, *qgb_orientation, *qgb_stackrgt/*, *qGroupBoxPatchConstraints*/;
	TYPE m_type = TYPE::FREE;
	QLabel *m_label = nullptr;
	QPushButton *m_addButton, *m_supprButton;
	QListWidget *m_listWidget;
	ProjectManagerWidget *m_projectManager = nullptr;
	std::vector<QString> m_names;
	std::vector<QString> m_path;
	bool m_multiSelection = false;
	void display();
	bool nameExist(std::vector<QString> names, QString name);
	WorkingSetManager *m_workingSetManager = nullptr;


	private slots:
	void trt_horizonAdd();
	void trt_horizonClear();

};





#endif
