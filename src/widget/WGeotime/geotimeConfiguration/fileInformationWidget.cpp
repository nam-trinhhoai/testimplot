

#include <QListWidget>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QString>
#include <QPushButton>
#include <QListWidget>
#include <QLineEdit>
#include <QComboBox>
#include <QLabel>
#include <QDialogButtonBox>
#include <QMessageBox>
#include <QFile>
#include <Xt.h>

#include "fileInformationWidget.h"

FileInformationWidget::FileInformationWidget(QString filename):QDialog()
{
	this->setWindowTitle("info");

	QVBoxLayout* pMainLayout = new QVBoxLayout();
	this->setLayout(pMainLayout);

	textInfo = new QPlainTextEdit();
	textInfo->setReadOnly(true);
	// textInfo->setMaximumBlockCount(1000);
	QDialogButtonBox *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok);
	pMainLayout->addWidget(textInfo);
	pMainLayout->addWidget(buttonBox);

	infoDisplay(filename);
	connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
}


FileInformationWidget::~FileInformationWidget()
{
}

/*
void FileInformationWidget::infoFromFilename(Q_OBJECT *parent, QString filename)
{
	QString info = infoFromFilename(filename);
	if ( !info.isEmpty() ) return;
	QMessageBox messageBox;
	messageBox.information(parent, "Info", info);
}
*/

QString FileInformationWidget::getFormatedFileSize(QString filename)
{
	double size = 0.0;
	QString fileSize = "";
	QFile file(filename);
	if ( file.open(QIODevice::ReadOnly) )
	{
		size = file.size();
		file.close();
	}
	if ( size < 1000 )
	{
		fileSize = QString::number((long)size) + " Bytes";
	}
	else if ( size < 1000000 )
	{
		fileSize = QString::number(size/1000.0, 'f', 2) + " kB";
	}
	else if ( size < 1000000000 )
	{
		fileSize = QString::number(size/1000000.0, 'f', 2) + " MB";
	}
	else
	{
		fileSize = QString::number(size/1000000000.0, 'f', 2) + " GB";
	}
	return fileSize;
}


QString FileInformationWidget::infoFromFilename(QString filename)
{
	QString info = "";

	if ( !QFile::exists(filename) )
		{

			return info;
		}

		double size = 0;
		QFile file(filename);
		if ( file.open(QIODevice::ReadOnly) )
		{
		    size = file.size();
		    file.close();
		}

		QFileInfo fi(filename);
		QString ext = fi.suffix();
		int dimx = 0;
		int dimy = 0;
		int dimz = 0;
		std::string xtType = "";

		if ( ext == "xt" || ext == "iso" )
		{
			inri::Xt xt(filename.toStdString());
			dimx = xt.nSamples();
			dimy = xt.nRecords();
			dimz = xt.nSlices();
			inri::Xt::Type type = xt.type();
			xtType = inri::Xt::type2str(type);
		}
		QString fileSize = "file size: ";
		if ( size < 1000 )
		{
			fileSize += QString::number((long)size) + " Bytes";
		}
		else if ( size < 1000000 )
		{
			fileSize += QString::number(size/1000.0, 'f', 2) + " kB";
		}
		else if ( size < 1000000000 )
		{
			fileSize += QString::number(size/1000000.0, 'f', 2) + " MB";
		}
		else
		{
			fileSize += QString::number(size/1000000000.0, 'f', 2) + " GB";
		}

		info += "location: " + filename;
		info += "\n";
		info += fileSize + "\n";
		if ( ext == "xt" || ext == "iso" )
		{
			info += "Type: " + QString::fromStdString(xtType) + "\n";
			info += "dimx: " + QString::number(dimx) + "\n";
			info += "dimy: " + QString::number(dimy) + "\n";
			info += "dimz: " + QString::number(dimz) + "\n";
		}
		return info;
}

void FileInformationWidget::infoDisplay(QString filename)
{
	if ( !QFile::exists(filename) )
	{

		return;
	}

	double size = 0;
	QFile file(filename);
	if ( file.open(QIODevice::ReadOnly) )
	{
	    size = file.size();
	    file.close();
	}

	QFileInfo fi(filename);
	QString ext = fi.suffix();
	int dimx = 0;
	int dimy = 0;
	int dimz = 0;
	std::string xtType = "";

	if ( ext == "xt" || ext == "iso" )
	{
		inri::Xt xt(filename.toStdString());
		dimx = xt.nSamples();
		dimy = xt.nRecords();
		dimz = xt.nSlices();
		inri::Xt::Type type = xt.type();
		xtType = inri::Xt::type2str(type);
	}
	QString fileSize = "file size: ";
	if ( size < 1000 )
	{
		fileSize += QString::number((long)size) + " Bytes";
	}
	else if ( size < 1000000 )
	{
		fileSize += QString::number(size/1000.0, 'f', 2) + " kB";
	}
	else if ( size < 1000000000 )
	{
		fileSize += QString::number(size/1000000.0, 'f', 2) + " MB";
	}
	else
	{
		fileSize += QString::number(size/1000000000.0, 'f', 2) + " GB";
	}


	this->textInfo->appendPlainText("location: " + filename);
	this->textInfo->appendPlainText(fileSize);
	if ( ext == "xt" || ext == "iso" )
	{
		this->textInfo->appendPlainText("Type: " + QString::fromStdString(xtType));
		this->textInfo->appendPlainText("dimx: " + QString::number(dimx));
		this->textInfo->appendPlainText("dimy: " + QString::number(dimy));
		this->textInfo->appendPlainText("dimz: " + QString::number(dimz));
	}
}


