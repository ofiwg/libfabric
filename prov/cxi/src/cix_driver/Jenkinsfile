@Library("dst-shared") _

obsBuildPipeline {
    product = "shs-cn,shs-ncn"
    productStream = "slingshot-host-software"
    timeout = 60
    numToKeepStr = 40
    recv_triggers = ["slingshot_base_link-built"]
    send_triggers = ["cxi-driver-built"]
    manifestGen = "true"
}

