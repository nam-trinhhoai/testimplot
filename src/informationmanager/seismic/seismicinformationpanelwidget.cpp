#include "seismicinformationpanelwidget.h"
#include "seismicinformation.h"

#include <QColorDialog>
#include <QFormLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>

SeismicInformationPanelWidget::SeismicInformationPanelWidget(SeismicInformation* information, QWidget* parent) :
		IInformationPanelWidget(parent), m_information(information) {
	QString name = "No name";
	QString size = "";
	QString sizeOnDisk = "";
	QString voxelFormat = "";
	QString axis = "";
	QString dataSetType = "";
	QString startSample = "";
	QString stepSample = "";
	QString startRecords = "";
	QString stepRecords = "";
	QString startSlices = "";
	QString stepSlices = "";
	if (!m_information.isNull()) {
		m_color = m_information->color();
		name = m_information->name();
		size = m_information->getSize();
		sizeOnDisk = m_information->getSizeOnDisk();
		voxelFormat = m_information->getVoxelFormat();
		axis = m_information->getAxis();
		dataSetType = m_information->getDataSetType();
		std::vector<QString> dataSetParams = m_information->getDataParams();
		if ( dataSetParams.size() == 6 )
		{
			startSample = dataSetParams[0];
			stepSample = dataSetParams[1];
			startRecords = dataSetParams[2];
			stepRecords = dataSetParams[3];
			startSlices = dataSetParams[4];
			stepSlices = dataSetParams[5];
		}
	}
	QFormLayout* mainLayout = new QFormLayout;
	setLayout(mainLayout);

	// QLineEdit *labelName = new QLineEdit(name);
	// labelName->setReadOnly(true);
	// labelName->setFrame(false);
	// labelName->setTextInteractionFlags(Qt::TextSelectableByMouse);
	QLineEdit *labelName = new QLineEdit(name);
	labelName->setReadOnly(true);
	labelName->setStyleSheet("QLineEdit { border: none }");
	mainLayout->addRow("Name: ", labelName);
	mainLayout->addRow("Data type: ", new QLabel(axis + " - [ " + dataSetType + " ]"));
	// mainLayout->addRow("Data: ", new QLabel(dataSetType));
	mainLayout->addRow("Size on disk: ", new QLabel(sizeOnDisk + " - [ " + size + " : " + voxelFormat + " ]"));
	// mainLayout->addRow("Voxel format: ", new QLabel(voxelFormat));
	// mainLayout->addRow("Dimensions: ", new QLabel(size));

	std::vector<float> dataSetFloatParam = m_information->getDataFloatParams();
	std::vector<int> dims = m_information->getDims();

	int vmin = (int)dataSetFloatParam[4];
	int step = (int)dataSetFloatParam[5];
	int vmax = vmin + step * (dims[2]-1);
	QHBoxLayout *zparam = infoDimensionCreate(vmin, vmax, step, "int", "dimz: " + QString::number(dims[2]));

	vmin = (int)dataSetFloatParam[2];
	step = (int)dataSetFloatParam[3];
	vmax = vmin + step * (dims[1]-1);
	QHBoxLayout *yparam = infoDimensionCreate(vmin, vmax, step, "int", "dimy: " + QString::number(dims[1]));

	float vminf = dataSetFloatParam[0];
	float stepf = dataSetFloatParam[1];
	float vmaxf = vminf + stepf * (dims[0]-1);
	QHBoxLayout *xparam = infoDimensionCreate(vminf, vmaxf, stepf, "float", "dimx: " + QString::number(dims[0]));

	QFrame *line1 = new QFrame;
	line1->setObjectName(QString::fromUtf8("line"));
	line1->setGeometry(QRect(320, 150, 118, 3));
	line1->setFrameShape(QFrame::HLine);
	mainLayout->addRow(line1);

	QHBoxLayout *dimHeader = new QHBoxLayout;
	QLabel *labelMin = new QLabel("Min:"); labelMin->setMaximumWidth(100);
	QLabel *labelMax = new QLabel("Max:"); labelMax->setMaximumWidth(100);
	QLabel *labelStep = new QLabel("Step:"); labelStep->setMaximumWidth(100);
	dimHeader->addWidget(labelMin);
	dimHeader->addWidget(labelMax);
	dimHeader->addWidget(labelStep);
	mainLayout->addRow("", dimHeader);
	mainLayout->addRow("inline:", zparam);
	mainLayout->addRow("xline:", yparam);
	QString labelSample = "Samples (m)";
	if ( axis == "Time" )
		labelSample = "Samples (ms)";
	mainLayout->addRow(labelSample, xparam);




	// =================================================================== // todo
	/*
	QHBoxLayout *qhb_zinfo = new QHBoxLayout;
	QLineEdit *qleZmin = new QLineEdit("1");
	qleZmin->setMaximumWidth(100);
	qleZmin->setReadOnly(true);
	qleZmin->setAlignment(Qt::AlignRight);
	QLineEdit *qleZmax = new QLineEdit("2");
	qleZmax->setMaximumWidth(100);
	qleZmax->setReadOnly(true);
	qleZmax->setAlignment(Qt::AlignRight);
	QLineEdit *qleZstep = new QLineEdit("3");
	qleZstep->setMaximumWidth(100);
	qleZstep->setReadOnly(true);
	qleZstep->setAlignment(Qt::AlignRight);
	qhb_zinfo->addWidget(qleZmin);
	qhb_zinfo->addWidget(qleZmax);
	qhb_zinfo->addWidget(qleZstep);

	QHBoxLayout *qhb_yinfo = new QHBoxLayout;
	QLineEdit *qleYmin = new QLineEdit("1");
	qleYmin->setMaximumWidth(100);
	qleYmin->setReadOnly(true);
	qleYmin->setAlignment(Qt::AlignRight);
	QLineEdit *qleYmax = new QLineEdit("2");
	qleYmax->setMaximumWidth(100);
	qleYmax->setReadOnly(true);
	qleYmax->setAlignment(Qt::AlignRight);
	QLineEdit *qleYstep = new QLineEdit("3");
	qleYstep->setMaximumWidth(100);
	qleYstep->setReadOnly(true);
	qleYstep->setAlignment(Qt::AlignRight);
	qhb_yinfo->addWidget(qleYmin);
	qhb_yinfo->addWidget(qleYmax);
	qhb_yinfo->addWidget(qleYstep);

	QHBoxLayout *qhb_xinfo = new QHBoxLayout;
	QLineEdit *qleXmin = new QLineEdit("1");
	qleXmin->setMaximumWidth(100);
	qleXmin->setReadOnly(true);
	qleXmin->setAlignment(Qt::AlignRight);
	QLineEdit *qleXmax = new QLineEdit("2");
	qleXmax->setMaximumWidth(100);
	qleXmax->setReadOnly(true);
	qleXmax->setAlignment(Qt::AlignRight);
	QLineEdit *qleXstep = new QLineEdit("3");
	qleXstep->setMaximumWidth(100);
	qleXstep->setReadOnly(true);
	qleXstep->setAlignment(Qt::AlignRight);
	qhb_xinfo->addWidget(qleXmin);
	qhb_xinfo->addWidget(qleXmax);
	qhb_xinfo->addWidget(qleXstep);
	*/

	/*
	float  startSample = dataSetParams[0];
		   stepSample = dataSetParams[1];
		   startRecords = dataSetParams[2];
				stepRecords = dataSetParams[3];
				startSlices = dataSetParams[4];
				stepSlices = dataSetParams[5];
	QHBoxLayout *qhb_zinfo = infoDimensionCreate(float vmin, float vmax, float step, QString format);

	mainLayout->addRow("inline:", qhb_zinfo);
	mainLayout->addRow("xline:", qhb_yinfo);
	mainLayout->addRow("Samples:", qhb_xinfo);
	*/











/*
	QString txt1 = "";
	QString txt2 = "";
	if ( axis == "Time")
	{
		txt1 = "first time: " + startSample;
		txt2 = "step time:  " + stepSample;
	}
	else if ( axis == "Depth" )
	{
		txt1 = "first depth: " + startSample;
		txt2 = "step depth:  " + stepSample;
	}
	mainLayout->addRow(txt1, new QLabel(txt2));
	// mainLayout->addRow("Start samples: ", new QLabel(startSample));
	// mainLayout->addRow("Step samples: ", new QLabel(stepSample));

	QFrame *line2 = new QFrame;
	line2->setObjectName(QString::fromUtf8("line"));
	line2->setGeometry(QRect(320, 150, 118, 3));
	line2->setFrameShape(QFrame::HLine);
	// mainLayout->addRow(line2);

	// mainLayout->addRow("Start records: ", new QLabel(startRecords));
	// mainLayout->addRow("Step records: ", new QLabel(stepRecords));
	txt1 = "first trace: " + startRecords;
	txt2 = "step trace:  " + stepRecords;
	mainLayout->addRow(txt1, new QLabel(txt2));


	QFrame *line3 = new QFrame;
	line3->setObjectName(QString::fromUtf8("line"));
	line3->setGeometry(QRect(320, 150, 118, 3));
	line3->setFrameShape(QFrame::HLine);
	// mainLayout->addRow(line3);
	txt1 = "first profil: " + startSlices;
	txt2 = "step profil:  " + stepSlices;
	mainLayout->addRow(txt1, new QLabel(txt2));
	*/

	// mainLayout->addRow("Start slices: ", new QLabel(startSlices));
	// mainLayout->addRow("Step slices: ", new QLabel(stepSlices));

	// connect(m_colorButton, &QPushButton::clicked, this, &NurbInformationPanelWidget::editColor);
}

SeismicInformationPanelWidget::~SeismicInformationPanelWidget() {

}

bool SeismicInformationPanelWidget::saveChanges() {
	bool valid = !m_information.isNull();
	if (valid) {
		m_information->setColor(m_color);
	}
	return valid;
}

QColor SeismicInformationPanelWidget::color() const {
	return m_color;
}

void SeismicInformationPanelWidget::setColor(QColor color) {
	m_color = color;
	setButtonColor(m_color);
}

void SeismicInformationPanelWidget::editColor() {
	QString name;
	if (!m_information.isNull()) {
		name = m_information->name();
	}

	QColor newColor = QColorDialog::getColor(m_color, this, "Select " + name + " color");
	if (newColor.isValid()) {
		setColor(newColor);
	}
}

void SeismicInformationPanelWidget::setButtonColor(const QColor& color) {
	// m_colorButton->setStyleSheet(QString("QPushButton{ background: %1; }").arg(m_color.name()));
}


QHBoxLayout *SeismicInformationPanelWidget::infoDimensionCreate(float vmin, float vmax, float step, QString format, QString suffix)
{
	QHBoxLayout *qhb_info = new QHBoxLayout;
	QLineEdit *qleMin = new QLineEdit();
	qleMin->setMaximumWidth(100);
	qleMin->setReadOnly(true);
	qleMin->setAlignment(Qt::AlignRight);
	QLineEdit *qleMax = new QLineEdit();
	qleMax->setMaximumWidth(100);
	qleMax->setReadOnly(true);
	qleMax->setAlignment(Qt::AlignRight);
	QLineEdit *qleStep = new QLineEdit();
	qleStep->setMaximumWidth(100);
	qleStep->setReadOnly(true);
	qleStep->setAlignment(Qt::AlignRight);
	qhb_info->addWidget(qleMin);
	qhb_info->addWidget(qleMax);
	qhb_info->addWidget(qleStep);
	if ( format == "int" )
	{
		qleMin->setText(QString::number((int)vmin));
		qleMax->setText(QString::number((int)vmax));
		qleStep->setText(QString::number((int)step));
	}
	else
	{
		qleMin->setText(QString::number(vmin, 'f', 4));
		qleMax->setText(QString::number(vmax, 'f', 4));
		qleStep->setText(QString::number(step, 'f', 4));
	}

	QLabel *label = new QLabel(suffix);
	qhb_info->addWidget(label);
	return qhb_info;
}

