#include "graphicsutil.h"
#include <QPushButton>

//#include <QDesktopWidget>
#include <QApplication>
#include <QScreen>
#include <iostream>

#include "slicerwindow.h"
#include "abstractgraphicsview.h"

QPushButton *GraphicsUtil::generateToobarButton(const QString&iconPath,const QString&tooltip,QWidget * parent )
{
	QPushButton *reset=new QPushButton(QIcon(iconPath),"",parent);
	reset->setToolTip(tooltip);
	reset->setStyleSheet("min-width: 32px;border: 0px solid;");
	reset->setFixedSize(24,24);
	reset->setIconSize(QSize(24, 24));
	reset->setDefault(false);
	reset->setAutoDefault(false);
	return reset;
}

void GraphicsUtil::arrangeWindows() {
	QList<QScreen*> screens = QGuiApplication::screens();
	QRect rec = screens.first()->availableVirtualGeometry();
	std::cout<<"Current Screen dims:"<<rec.x()<<"\t"<<rec.y()<<"\t"<<rec.width()<<"\t"<<rec.height()<<std::endl;
	int height = rec.height();
	int width = rec.width();
	if (SlicerWindow::get()==nullptr) {
		return;
	}
	QVector<AbstractGraphicsView*> currentViewerList =
			SlicerWindow::get()->currentViewerList();
	std::cout<<"Visible windows:"<<currentViewerList.size()<<std::endl;
	int nbLines = 2;
	if (currentViewerList.size() <= 2)
		nbLines = 1;

	int numWidth = currentViewerList.size() / nbLines;
	if(currentViewerList.size()%nbLines==1)
		numWidth+=1;

	int targetWidth = width / numWidth;
	int targetHeight=height / nbLines;
	int viewIndex = 0;
	int topX = 0, topY = 0;
	for (int j = 0; j < nbLines; j++) {
		//Compute tthe residual dims
		if (j == nbLines - 1)
			targetWidth = width / (currentViewerList.size() - viewIndex);

		for (int i = 0; i < numWidth; i++) {
			currentViewerList[viewIndex]->setGeometry(rec.x() + topX,
					rec.y() + topY, targetWidth, targetHeight);
			topX += targetWidth;
			viewIndex++;
			if (viewIndex == currentViewerList.size() )
				break;
		}
		topY+=targetHeight;
		topX=0;
		if (viewIndex == currentViewerList.size())
			break;

	}

}
