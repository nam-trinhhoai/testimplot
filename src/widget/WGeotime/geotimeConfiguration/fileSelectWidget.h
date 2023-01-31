
#ifndef __FILESELECTWIDGET__
#define __FILESELECTWIDGET__

class QPushButton;
class QLineEdit;
class QLabel;
class QString;
class QPoint;

class ProjectManagerWidget;
class WorkingSetManager;


class FileSelectWidget : public QWidget{
	Q_OBJECT
public:
	enum FILE_SORT_TYPE { All = 0, Seismic, dip, dipxy, dipxz, Rgt, patch, Raw, Avi, Other };
	enum FILE_FORMAT { ALL, INT16, UINT32, FLOAT32 };
	enum FILE_TYPE { seismic, rgtCubeToAttribut, avi };
	FileSelectWidget(QWidget* parent = 0);
	FileSelectWidget(QString label, QString buttonLabel, QString lineEditLabel,  QWidget* parent = 0);
	virtual ~FileSelectWidget();
	void ihmCreate();
	void setProjectManager(ProjectManagerWidget *manager);
	void setLabelText(QString txt);
	void setButtonText(QString txt);
	void setLineEditText(QString txt);
	void setFileSortType(int val);
	void setFileType(FILE_TYPE type);
	void setReadOnly(bool val);
	void setLabelDimensionVisible(bool val);
	void setWorkingSetManager(WorkingSetManager *p) { m_workingSetManager = p; }
	QString getLineEditText();
	QString getFilename();
	QString getPath();
	void setDims(int dimx, int dimy, int dimz);
	void clear();
	void setFileFormat(FILE_FORMAT format) { m_fileFormat = format; }


private:
	// QGroupBox* qgb_seismic, *qgb_orientation, *qgb_stackrgt/*, *qGroupBoxPatchConstraints*/;
	QPushButton *m_button = nullptr;
	QLineEdit *m_lineEdit = nullptr;
	QLabel *m_label = nullptr;
	QLabel *m_labelDimensions = nullptr;
	ProjectManagerWidget *m_projectManager = nullptr;
	WorkingSetManager *m_workingSetManager = nullptr;
	QString filename = "";
	QString path = "";
	int fileSortType = FILE_SORT_TYPE::All;
	FILE_TYPE fileType = FILE_TYPE::seismic;
	void updateLabelDimensions(QString filename);
	void ProvideContextMenu(QLineEdit *lineEdit, const QPoint &pos);
	void seismicFileOpen();
	bool fileFormatCheck(QString path);
	QString fileFormatString(FILE_FORMAT format);


	int dimx0 = -1;
	int dimy0 = -1;
	int dimz0 = -1;
	QString m_textOriginal = "";
	FILE_FORMAT m_fileFormat = FILE_FORMAT::INT16;

signals:
	void filenameChanged();


private slots:
	void buttonClick();
	void ProvideContextMenuList(const QPoint &pos);


};













#endif
