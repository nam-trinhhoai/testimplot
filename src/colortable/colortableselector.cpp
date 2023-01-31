#include "colortableselector.h"
#include "colortablechooser.h"

#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPainter>
#include <QPushButton>

#include <QDebug>

ColorTableSelector::ColorTableSelector(QWidget* parent) : QWidget(parent)
{

	QHBoxLayout *mainLayout=new QHBoxLayout(this);
	mainLayout->setSpacing(0);
	mainLayout->setContentsMargins(0, 0, 0, 0);

	m_iconResult = new QLabel;
	m_iconResult->setFixedSize(30,16);
	m_iconResult->setMaximumSize(QSize(30,16));
	m_iconResult->setStyleSheet("min-width: 30px;");
	mainLayout->addWidget(m_iconResult);

    m_result = new QLineEdit();
    mainLayout->addWidget(m_result);
    m_openDlg = new QPushButton("...");
    m_openDlg->setFixedSize(25,24);
    m_openDlg->setMaximumSize(QSize(25,24));
    m_openDlg->setStyleSheet("min-width: 25px;");
    m_openDlg->setDefault(false);
    m_openDlg->setAutoDefault(false);
    mainLayout->addWidget(m_openDlg);
    m_chooser=nullptr;

    connect(m_openDlg, &QPushButton::clicked,
    		this, &ColorTableSelector::openChooser);
 }

ColorTableSelector::~ColorTableSelector()
{
	delete m_chooser;
}

void ColorTableSelector::openChooser()
{
	//m_openDlg->setEnabled(false);
	if(!m_chooser) {
		m_chooser = new ColorTableChooser(this);
	}
	m_chooser->show();
	//m_openDlg->setEnabled(true);
}

void ColorTableSelector::changeSelection(const ColorTable& newSelection)
{
	qDebug() << "ColorTableSelector::changeSelection called";
    if(m_curSelection == newSelection)
        return;

    m_curSelection= newSelection;
    m_result->setText( newSelection.getName().c_str() );
    m_iconResult->setPixmap(getPixmap(newSelection));
    emit selectionChanged(m_curSelection);
}

void ColorTableSelector::setSelection(const ColorTable& newSelection)
{

    if(m_curSelection == newSelection)
        return;

    m_curSelection= newSelection;
    m_result->setText(newSelection.getName().c_str());
    m_iconResult->setPixmap(getPixmap(newSelection));

    if(m_chooser) {
        m_chooser->selectLineFromData(newSelection);
    }
}

const ColorTable& ColorTableSelector::getCurrentSelection() const {
    return m_curSelection;
}

QPixmap ColorTableSelector::getPixmap(const ColorTable& colorTable) {
    QImage img(colorTable.size(), 16, QImage::Format_RGB32);
    QPainter p(&img);
    p.fillRect(img.rect(), Qt::black);

    for (int i = 0; i < colorTable.size(); i++) {
        QRect rect = img.rect().adjusted(i, 1, i + 1, 14);
        const std::array<int, 4> color = colorTable.getColors(i);
        p.fillRect(rect,QColor(color[0], color[1], color[2], color[3]));
    }

    return QPixmap::fromImage(img.scaled(30, 16));
}
