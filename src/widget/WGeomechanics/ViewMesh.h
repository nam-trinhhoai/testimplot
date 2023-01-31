/*
 *
 *
 *  Created on: 27 May 2022
 *      Author: l0359127
 */

#ifndef NEXTVISION_SRC_WIDGET_WGEOMECHANICS_VIEWMESH_H_
#define NEXTVISION_SRC_WIDGET_WGEOMECHANICS_VIEWMESH_H_

#include<QWidget>
#include<Qt3DExtras>
#include<QVBoxLayout>


class ViewMesh : public QWidget
{
	Q_OBJECT

public:
    ViewMesh(QWidget *parent=nullptr);
    virtual ~ViewMesh();

private:

	Qt3DExtras::Qt3DWindow *m_view;
	Qt3DCore::QEntity *createScene();
private slots:

};


#endif // NEXTVISION_SRC_WIDGET_WGEOMECHANICS_VIEWMESH_H_ 

