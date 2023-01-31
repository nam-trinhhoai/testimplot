

#ifndef __LABELLINEEDITWIDGET__
#define __LABELLINEEDITWIDGET__


class QLineEdit;
class QLabel;
class QString;



class LabelLineEditWidget : public QWidget{
	Q_OBJECT
public:
	LabelLineEditWidget(QWidget* parent = 0);
	LabelLineEditWidget(QString label, QString lineEditLabel,  QWidget* parent = 0);
	virtual ~LabelLineEditWidget();
	void setLabelText(QString txt);
	void setLineEditText(QString txt);
	QString getLineEditText();


private:
	// QGroupBox* qgb_seismic, *qgb_orientation, *qgb_stackrgt/*, *qGroupBoxPatchConstraints*/;
	QLineEdit *m_lineEdit = nullptr;
	QLabel *m_label = nullptr;

};


#endif
