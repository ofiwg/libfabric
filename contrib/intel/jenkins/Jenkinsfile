
pipeline {
    agent any
    options {timestamps()}
    /*triggers {
        pollSCM('H/2 * * * *')
    } */
    stages {
        stage ('fetch-opa-psm2')  {
             steps {
                 withEnv(['PATH+EXTRA=/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/bin']) { 
                     dir('opa-psm2-lib') {

                        checkout changelog: false, poll: false, scm: [$class: 'GitSCM', \
                        branches: [[name: '*/master']], \
                        doGenerateSubmoduleConfigurations: false, extensions: [], submoduleCfg: [], \
                        userRemoteConfigs: [[url: 'https://github.com/intel/opa-psm2.git']]]                        
                      }
                 }
             }
        }
        
        stage ('build-libfabric') {
            steps {
                withEnv(['PATH+EXTRA=/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/bin']) { 
                sh """
                  python3.7 contrib/intel/jenkins/build.py 'libfabric'
                  python3.7 contrib/intel/jenkins/build.py 'libfabric' --ofi_build_mode='dbg'
                  python3.7 contrib/intel/jenkins/build.py 'libfabric' --ofi_build_mode='dl'
                  echo "libfabric build completed"  
                 """
                }
            }
        }
        stage('build-fabtests') {
            steps {
                withEnv(['PATH+EXTRA=/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/bin']) { 
                sh """
                python3.7 contrib/intel/jenkins/build.py 'fabtests'
                python3.7 contrib/intel/jenkins/build.py 'fabtests' --ofi_build_mode='dbg'
                python3.7 contrib/intel/jenkins/build.py 'fabtests' --ofi_build_mode='dl'              
                echo 'fabtests build completed' 
                """
                }
            }
        }
        
        stage ('build-shmem') {
            steps {
              withEnv(['PATH+EXTRA=/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/bin']) {
                sh """
                python3.7  contrib/intel/jenkins/build.py 'shmem'
                echo 'shmem benchmarks built successfully'
                """
                }
              }
          }
  
        stage ('build OMPI_bm') {
              steps {
              withEnv(['PATH+EXTRA=/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/bin']) {
                  sh """
                  python3.7 contrib/intel/jenkins/build.py 'ompi_benchmarks' 
                  python3.7 contrib/intel/jenkins/build.py 'ompi_benchmarks' --ofi_build_mode='dbg' 
                  python3.7 contrib/intel/jenkins/build.py 'ompi_benchmarks' --ofi_build_mode='dl'
                  echo 'mpi benchmarks with ompi - built successfully'
                 """
                }
              }
          }
    
    stage('build IMPI_bm') {
        steps {
          withEnv(['PATH+EXTRA=/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/bin']) {
                sh """
                python3.7 contrib/intel/jenkins/build.py 'impi_benchmarks'
                python3.7 contrib/intel/jenkins/build.py 'impi_benchmarks' --ofi_build_mode='dbg'
                python3.7 contrib/intel/jenkins/build.py 'impi_benchmarks' --ofi_build_mode='dl'
                echo 'mpi benchmarks with impi - built successfully'
                """
            }
          }
      }  
    
    stage('build MPICH_bm') {
        steps {
          withEnv(['PATH+EXTRA=/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/bin']) {
                sh """
                python3.7 contrib/intel/jenkins/build.py 'mpich_benchmarks'
                python3.7 contrib/intel/jenkins/build.py 'mpich_benchmarks' --ofi_build_mode='dbg'
                python3.7 contrib/intel/jenkins/build.py 'mpich_benchmarks' --ofi_build_mode='dl'
                echo "mpi benchmarks with mpich - built successfully"
                """
              }
            }
        }
   stage('parallel-tests') {
            parallel {
                stage('eth-test') {
                     agent {node {label 'eth'}}
                     steps{
                        withEnv(['PATH+EXTRA=/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/bin/:$PYTHONPATH'])
                        {
                          sh """
                            env
                            (
                                cd  ${env.WORKSPACE}/contrib/intel/jenkins/ 
                                python3.7 runtests.py --prov=tcp
                                python3.7 runtests.py --prov=udp 
                                python3.7 runtests.py --prov=sockets               
                            )                              
                          """
                        }
                     }
                 }
                 stage('hfi1-test') {
                     agent {node {label 'hfi1'}}
                     steps{
                        withEnv(['PATH+EXTRA=/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/bin:$PYTHONPATH']) {
                          sh """
                            env
                            (
                                cd ${env.WORKSPACE}/contrib/intel/jenkins/
                                python3.7 runtests.py --prov=psm2
                                python3.7 runtests.py --prov=verbs                   
                            )
                          """
                        } 
                     }       
       
                 }
                 stage('mlx5-test') {
                     agent {node {label 'mlx5'}}
                     steps{
                        withEnv(['PATH+EXTRA=/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/bin:$PYTHONPATH']) {
                          sh """
                            env
                            (                            
                                cd ${env.WORKSPACE}/contrib/intel/jenkins/
                                python3.7 runtests.py --prov=verbs                   
                            )  
                          """
                        } 
                     }       
       
                 }
                 stage('eth-test-dbg') {
                     agent {node {label 'eth'}}
                     steps{
                        withEnv(['PATH+EXTRA=/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/bin/:$PYTHONPATH'])
                        {
                          sh """
                            env
                            ( 
                                cd  ${env.WORKSPACE}/contrib/intel/jenkins/
                                python3.7 runtests.py --prov=tcp --ofi_build_mode='dbg'
                                python3.7 runtests.py --prov=udp --ofi_build_mode='dbg'
                                python3.7 runtests.py --prov=sockets --ofi_build_mode='dbg'               
                            )  
                          """
                        } 
                     }       
       
                 }
                 stage('hfi1-test-dbg') {
                     agent {node {label 'hfi1'}}
                     steps{
                        withEnv(['PATH+EXTRA=/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/bin:$PYTHONPATH']) {
                          sh """
                            env 
                            (
                                cd ${env.WORKSPACE}/contrib/intel/jenkins/
                                python3.7 runtests.py --prov=psm2 --ofi_build_mode='dbg'
                                python3.7 runtests.py --prov=verbs --ofi_build_mode='dbg'                   
                            ) 
                         """
                        } 
                     }       
       
                 }
                 stage('mlx5-test-dbg') {
                     agent {node {label 'mlx5'}}
                     steps{
                        withEnv(['PATH+EXTRA=/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/bin:$PYTHONPATH']) {
                          sh """
                            env
                            (
                                cd ${env.WORKSPACE}/contrib/intel/jenkins/
                                python3.7 runtests.py --prov=verbs --ofi_build_mode='dbg'                   
                            ) 
                          """
                        } 
                     }       
       
                 }
                 stage('eth-test-dl') {
                     agent {node {label 'eth'}}
                     steps{
                        withEnv(['PATH+EXTRA=/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/bin/:$PYTHONPATH'])
                        {
                          sh """
                            env
                            (
                                cd  ${env.WORKSPACE}/contrib/intel/jenkins/
                                python3.7 runtests.py --prov=tcp --ofi_build_mode='dl'
                                python3.7 runtests.py --prov=udp --ofi_build_mode='dl'
                                python3.7 runtests.py --prov=sockets --ofi_build_mode='dl'               
                            )  
                        """
                        } 
                     }       
       
                 }
                 
                 stage('hfi1-test-dl') {
                     agent {node {label 'hfi1'}}
                     steps{
                        withEnv(['PATH+EXTRA=/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/bin:$PYTHONPATH']) {
                          sh """
                            env
                            ( 
                                cd ${env.WORKSPACE}/contrib/intel/jenkins/
                                python3.7 runtests.py --prov=psm2 --ofi_build_mode='dl'
                                python3.7 runtests.py --prov=verbs --ofi_build_mode='dl'                   
                            ) 
                         """
                        } 
                     }       
       
                 }
                 
                 stage('mlx5-test-dl') {
                     agent {node {label 'mlx5'}}
                     steps{
                        withEnv(['PATH+EXTRA=/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/bin:$PYTHONPATH']) {
                          sh """
                            env
                            (                           
                                cd ${env.WORKSPACE}/contrib/intel/jenkins/
                                python3.7 runtests.py --prov=verbs --ofi_build_mode='dl'                   
                            )  
                         """
                        } 
                     }       
       
                 }   
    
            } 
   }        

  }
}
