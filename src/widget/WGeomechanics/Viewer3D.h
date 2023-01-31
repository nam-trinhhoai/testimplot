
/**
 *
 *
 *  Created on: 27 May 2022
 *      Author: l0359127
 */


#ifndef NEXTVISION_SRC_WIDGET_WGEOMECHANICS_VIEWER3D_H_
#define NEXTVISION_SRC_WIDGET_WGEOMECHANICS_VIEWER3D_H_
/**
#include <iostream>
#include <fstream>

#include <QSplitter>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QTextEdit>
#include <QLineEdit>
#include <QFile>
#include <QString>
#include <QStringList>
#include <QLabel>
#include <QFileDialog>
#include <QTextStream>
#include <QVector>
#include <QVector3D>
*/
//#include <Qt3DExtras>
//#include <QDialog>

#include<QWidget>
class Viewer3D : public QWidget
{
    Q_OBJECT

    public:
        Viewer3D(QWidget *parent = nullptr);
		virtual ~Viewer3D();
//        SceneModifier* sceneModifier() {return m_sceneModifier;}
/**
    protected:
        bool eventFilter(QObject *obj, QEvent *ev);
        void mouseMoveEvent(QMouseEvent *ev);
        void mousePressEvent(QMouseEvent *ev);
        void mouseReleaseEvent(QMouseEvent *ev);
        void wheelEvent(QWheelEvent *we);
*/
    private:

//        QPointer<Qt3DCore::QEntity> m_rootEntity;
//        QPointer<SceneModifier> m_sceneModifier;
//        Qt3DExtras::Qt3DWindow *m_view;
//        QPoint m_moveStartPoint;
//        QMatrix4x4 m_cameraMatrix;
};

#endif // NEXTVISION_SRC_WIDGET_WGEOMECHANICS_VIEWER3D_H_
